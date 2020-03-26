#ifndef FGC_JSD_H__
#define FGC_JSD_H__
#include "fgc/distance.h"
#include "distmat/distmat.h"
#include "fgc/kmeans.h"
#include "fgc/coreset.h"
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/polygamma.hpp>


namespace fgc {

namespace jsd {

using namespace blz;
using namespace blz::distance;


enum ProbDivType {
    L1,
    L2,
    SQRL2,
    JSM, // Multinomial Jensen-Shannon Metric
    JSD, // Multinomial Jensen-Shannon Divergence
    MKL, // Multinomial KL Divergence
    POISSON, // Poisson KL
    HELLINGER,
    BHATTACHARYA_METRIC,
    BHATTACHARYA_DISTANCE,
    TOTAL_VARIATION_DISTANCE,
    LLR,
    EMD,
    REVERSE_MKL,
    REVERSE_POISSON,
    UWLLR, /* Unweighted Log-likelihood Ratio.
            * Specifically, this is the D_{JSD}^{\lambda}(x, y),
            * where \lambda = \frac{N_p}{N_p + N_q}
            *
            */
    WLLR = LLR, // Weighted Log-likelihood Ratio, now equivalent to the LLR
    TVD = TOTAL_VARIATION_DISTANCE,
    WASSERSTEIN=EMD,
    PSD = JSD, // Poisson JSD, but algebraically equivalent
    PSM = JSM,
};

namespace detail {
static INLINE bool  needs_logs(ProbDivType d)  {
    switch(d) {
        case JSM: case JSD: case MKL: case POISSON: case LLR:
        case REVERSE_MKL: case REVERSE_POISSON: case UWLLR: return true;
        default: break;
    }
    return false;
}

static INLINE bool  needs_sqrt(ProbDivType d) {
    return d == HELLINGER || d == BHATTACHARYA_METRIC || d == BHATTACHARYA_DISTANCE;
}

static INLINE bool is_symmetric(ProbDivType d) {
    switch(d) {
        case L1: case L2: case EMD: case HELLINGER: case BHATTACHARYA_DISTANCE: case BHATTACHARYA_METRIC:
        case JSD: case JSM: case LLR: case UWLLR: case SQRL2: case TOTAL_VARIATION_DISTANCE:
            return true;
        default: ;
    }
    return false;
}

static INLINE const char *prob2str(ProbDivType d) {
    switch(d) {
        case BHATTACHARYA_DISTANCE: return "BHATTACHARYA_DISTANCE";
        case BHATTACHARYA_METRIC: return "BHATTACHARYA_METRIC";
        case EMD: return "EMD";
        case HELLINGER: return "HELLINGER";
        case JSD: return "JSD/PSD";
        case JSM: return "JSM/PSM";
        case L1: return "L1";
        case L2: return "L2";
        case LLR: return "LLR";
        case UWLLR: return "UWLLR";
        case MKL: return "MKL";
        case POISSON: return "POISSON";
        case REVERSE_MKL: return "REVERSE_MKL";
        case REVERSE_POISSON: return "REVERSE_POISSON";
        case SQRL2: return "SQRL2";
        case TOTAL_VARIATION_DISTANCE: return "TOTAL_VARIATION_DISTANCE";
        default: return "INVALID TYPE";
    }
}
} // detail

template<typename MatrixType>
class ProbDivApplicator {
    //using opposite_type = typename base_type::OppositeType;
    MatrixType &data_;
    using VecT = blaze::DynamicVector<typename MatrixType::ElementType>;
    std::unique_ptr<VecT> row_sums_;
    std::unique_ptr<MatrixType> logdata_;
    std::unique_ptr<MatrixType> sqrdata_;
    std::unique_ptr<VecT> llr_cache_, jsdcache_;
    typename MatrixType::ElementType lambda_ = 0.5;
public:
    using FT = typename MatrixType::ElementType;
    using MT = MatrixType;
    using This = ProbDivApplicator<MatrixType>;
    using ConstThis = const ProbDivApplicator<MatrixType>;

    const ProbDivType measure_;
    const MatrixType &data() const {return data_;}
    size_t size() const {return data_.rows();}
    template<typename PriorContainer=blaze::DynamicVector<FT, blaze::rowVector>>
    ProbDivApplicator(MatrixType &ref,
                      ProbDivType measure=JSM,
                      Prior prior=NONE,
                      const PriorContainer *c=nullptr):
        data_(ref), logdata_(nullptr), measure_(measure)
    {
        prep(prior, c);
    }
    /*
     * Sets distance matrix, under measure_ (if not provided)
     * or measure (if provided as an argument).
     */
    template<typename MatType>
    void set_distance_matrix(MatType &m, bool symmetrize=false) const {set_distance_matrix(m, measure_, symmetrize);}

    template<typename MatType>
    void set_distance_matrix(MatType &m, ProbDivType measure, bool symmetrize=false) const {
        using blaze::sqrt;
        const size_t nr = m.rows();
        assert(nr == m.columns());
        assert(nr == data_.rows());
        ProbDivType actual_measure = measure == JSM ? JSD: measure;
        for(size_t i = 0; i < nr; ++i) {
            CONST_IF((blaze::IsDenseMatrix_v<MatrixType>)) {
                for(size_t j = i + 1; j < nr; ++j) {
                    auto v = this->operator()(i, j, actual_measure);
                    m(i, j) = v;
                }
            } else {
                OMP_PFOR
                for(size_t j = i + 1; j < nr; ++j) {
                    auto v = this->operator()(i, j, actual_measure);
                    m(i, j) = v;
                }
            }
        }
        if(measure == JSM) {
            CONST_IF(blaze::IsDenseMatrix_v<MatType> || blaze::IsSparseMatrix_v<MatType>) {
                m = blz::sqrt(m);
            } else CONST_IF(dm::is_distance_matrix_v<MatType>) {
                blaze::CustomVector<FT, blaze::unaligned, blaze::unpadded> cv(const_cast<FT *>(m.data()), m.size());
                cv = blz::sqrt(cv);
            } else {
                std::transform(m.begin(), m.end(), m.begin(), [](auto x) {return std::sqrt(x);});
            }
        }
        if(detail::is_symmetric(measure)) {
            if(symmetrize) {
                fill_symmetric_upper_triangular(m);
            }
        } else {
            CONST_IF(dm::is_distance_matrix_v<MatType>) {
                std::fprintf(stderr, "Warning: using asymmetric measure with an upper triangular matrix. You are computing only half the values");
            } else {
                for(size_t i = 1; i < nr; ++i) {
                    CONST_IF((blaze::IsDenseMatrix_v<MatrixType>)) {
                        for(size_t j = 0; j < i; ++j) {
                            auto v = this->operator()(i, j, measure);
                            m(i, j) = v;
                        }
                    } else {
                        OMP_PFOR
                        for(size_t j = 0; j < i; ++j) {
                            auto v = this->operator()(i, j, measure);
                            m(i, j) = v;
                        }
                    }
                    m(i, i) = 0.;
                }
            }
        }
    }
    blaze::DynamicMatrix<float> make_distance_matrix() const {
        blaze::DynamicMatrix<float> ret = make_distance_matrix(measure_);
#if FGC_DBG_PRINT
        std::cout << ret << '\n';
#endif
        return ret;
    }
    blaze::DynamicMatrix<float> make_distance_matrix(ProbDivType measure, bool symmetrize=false) const {
        std::fprintf(stderr, "About to make distance matrix of %zu/%zu with %s calculated\n", data_.rows(), data_.rows(), detail::prob2str(measure));
        blaze::DynamicMatrix<float> ret(data_.rows(), data_.rows());
        set_distance_matrix(ret, measure, symmetrize);
        return ret;
    }
    // Accessors
    auto weighted_row(size_t ind) const {
        return blz::row(data_, ind BLAZE_CHECK_DEBUG) * row_sums_->operator[](ind);
    }
    auto row(size_t ind) const {return blz::row(data_, ind BLAZE_CHECK_DEBUG);}
    auto logrow(size_t ind) const {return blz::row(*logdata_, ind BLAZE_CHECK_DEBUG);}
    auto sqrtrow(size_t ind) const {return blz::row(*sqrdata_, ind BLAZE_CHECK_DEBUG);}


    /*
     * Distances
     */
    INLINE double operator()(size_t i, size_t j) const {
        return this->operator()(i, j, measure_);
    }
    INLINE double operator()(size_t i, size_t j, ProbDivType measure) const {
        if(unlikely(i >= data_.rows() || j >= data_.rows())) {
            std::cout << (std::string("Invalid rows selection: ") + std::to_string(i) + ", " + std::to_string(j) + '\n');
            std::exit(1);
        }
        double ret;
        switch(measure) {
            case TOTAL_VARIATION_DISTANCE: ret = discrete_total_variation_distance(row(i), row(j)); break;
            case L1:    ret = l1Norm(weighted_row(i) - weighted_row(j)); break;
            case L2:    ret = l2Norm(weighted_row(i) - weighted_row(j)); break;
            case SQRL2: ret = blaze::sqrNorm(weighted_row(i) - weighted_row(j)); break;
            case JSD:   ret = jsd(i, j); break;
            case JSM:   ret = jsm(i, j); break;
            case REVERSE_MKL: std::swap(i, j); [[fallthrough]];
            case MKL:   ret = mkl(i, j); break;
            case EMD:   ret = p_wasserstein(row(i), row(j)); break;
            case REVERSE_POISSON: std::swap(i, j); [[fallthrough]];
            case POISSON: ret = pkl(i, j); break;
            case HELLINGER: ret = hellinger(i, j); break;
            case BHATTACHARYA_METRIC: ret = bhattacharya_metric(i, j); break;
            case BHATTACHARYA_DISTANCE: ret = bhattacharya_distance(i, j); break;
            case LLR: ret = llr(i, j); break;
            case UWLLR: ret = uwllr(i, j); break;
            default: __builtin_unreachable();
        }
        return ret;
    }
    template<typename MatType>
    void operator()(MatType &mat, ProbDivType measure, bool symmetrize=false) {
        set_distance_matrix(mat, measure, symmetrize);
    }
    template<typename MatType>
    void operator()(MatType &mat, bool symmetrize=false) {
        set_distance_matrix(mat, symmetrize);
    }
    template<typename MatType>
    auto operator()() {
        return make_distance_matrix(measure_);
    }

    double hellinger(size_t i, size_t j) const {
        return sqrdata_ ? blaze::sqrNorm(sqrtrow(i) - sqrtrow(j))
                        : blaze::sqrNorm(blz::sqrt(row(i)) - blz::sqrt(row(j)));
    }
    double jsd(size_t i, size_t j) const {
        assert(i < data_.rows());
        assert(j < data_.rows());
        double ret;
        auto ri = row(i), rj = row(j);
        //constexpr FT logp5 = -0.693147180559945; // std::log(0.5)
        auto s = ri + rj;
        ret = jsdcache_->operator[](i) + jsdcache_->operator[](j) - blz::dot(s, blaze::neginf2zero(blaze::log(s * 0.5)));
#ifndef NDEBUG
        static constexpr typename MatrixType::ElementType threshold
            = std::is_same_v<typename MatrixType::ElementType, double>
                                ? 0.: -1e-5;
        assert(ret >= threshold || !std::fprintf(stderr, "ret: %g (numerical stability issues)\n", ret));
#endif
        return std::max(ret, 0.);
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    double jsd(size_t i, const OT &o, const OT2 &olog) const {
        auto mnlog = evaluate(log(0.5 * (row(i) + o)));
        return (blz::dot(row(i), logrow(i) - mnlog) + blz::dot(o, olog - mnlog));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    double jsd(size_t i, const OT &o) const {
        auto olog = evaluate(blaze::neginf2zero(blz::log(o)));
        return jsd(i, o, olog);
    }
    double mkl(size_t i, size_t j) const {
        // Multinomial KL
        return (*jsdcache_)[i] - blz::dot(row(i), logrow(j));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    double mkl(size_t i, const OT &o) const {
        // Multinomial KL
        return (*jsdcache_)[i] - blz::dot(row(i), blaze::neginf2zero(blz::log(o)));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    double mkl(size_t i, const OT &, const OT2 &olog) const {
        // Multinomial KL
        return blz::dot(row(i), logrow(i) - olog);
    }
    double pkl(size_t i, size_t j) const {
        // Poission KL
        return blz::dot(row(i), logrow(i) - logrow(j)) + blz::sum(row(j) - row(i));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    double pkl(size_t i, const OT &o, const OT2 &olog) const {
        // Poission KL
        return blz::dot(row(i), logrow(i) - olog) + blz::sum(row(i) - o);
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    double pkl(size_t i, const OT &o) const {
        return pkl(i, o, neginf2zero(blz::log(o)));
    }
    double psd(size_t i, size_t j) const {
        // Poission JSD
        auto mnlog = evaluate(log(.5 * (row(i) + row(j))));
        return (blz::dot(row(i), logrow(i) - mnlog) + blz::dot(row(j), logrow(j) - mnlog));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    double psd(size_t i, const OT &o, const OT2 &olog) const {
        // Poission JSD
        auto mnlog = evaluate(log(.5 * (row(i) + o)));
        return (blz::dot(row(i), logrow(i) - mnlog) + blz::dot(o, olog - mnlog));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    double psd(size_t i, const OT &o) const {
        return psd(i, o, neginf2zero(blz::log(o)));
    }
    double bhattacharya_sim(size_t i, size_t j) const {
        return sqrdata_ ? blz::dot(sqrtrow(i), sqrtrow(j))
                        : blz::sum(blz::sqrt(row(i) * row(j)));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    double bhattacharya_sim(size_t i, const OT &o, const OT2 &osqrt) const {
        return sqrdata_ ? blz::dot(sqrtrow(i), osqrt)
                        : blz::sum(blz::sqrt(row(i) * o));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    double bhattacharya_sim(size_t i, const OT &o) const {
        return bhattacharya_sim(i, o, blz::sqrt(o));
    }
    template<typename...Args>
    double bhattacharya_distance(Args &&...args) const {
        return -std::log(bhattacharya_sim(std::forward<Args>(args)...));
    }
    template<typename...Args>
    double bhattacharya_metric(Args &&...args) const {
        return std::sqrt(1 - bhattacharya_sim(std::forward<Args>(args)...));
    }
    template<typename...Args>
    double psm(Args &&...args) const {return std::sqrt(std::forward<Args>(args)...);}
    auto llr(size_t i, size_t j) const {
            //blaze::dot(row(i), logrow(i)) * row_sums_->operator[](i)
            //+
            //blaze::dot(row(j), logrow(j)) * row_sums_->operator[](j)
            // X_j^Tlog(p_j)
            // X_k^Tlog(p_k)
            // (X_k + X_j)^Tlog(p_jk)
        const auto lhn = (*llr_cache_)[i], rhn = (*llr_cache_)[j];
        const auto lambda = lhn / (lhn + rhn), m1l = 1. - lambda;
        auto ret = lambda * lhn + m1l * rhn
            -
            blz::dot(weighted_row(i) + weighted_row(j),
                neginf2zero(blz::log(lambda * row(i) + m1l * row(j)))
            );
        assert(ret >= -1e-2 * (row_sums_->operator[](i) + row_sums_->operator[](j)) || !std::fprintf(stderr, "ret: %g\n", ret));
        return std::max(ret, 0.);
    }
    double uwllr(size_t i, size_t j) const {
        const auto lhn = (*llr_cache_)[i], rhn = (*llr_cache_)[j];
        const double lambda = lhn / (lhn + rhn), m1l = 1. - lambda;
        return
          std::max(
            lambda * (*jsdcache_)[i] +
                  m1l * (*jsdcache_)[j] -
               blz::dot(lambda * row(i) + m1l * row(j),
                        neginf2zero(blz::log(
                            lambda * row(i) + m1l * row(j)))),
          0.);
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    double llr(size_t, const OT &) const {
        throw std::runtime_error("llr is not implemented for this.");
        return 0.;
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    double llr(size_t, const OT &, const OT2 &) const {
        throw std::runtime_error("llr is not implemented for this.");
        return 0.;
    }
    template<typename...Args>
    double jsm(Args &&...args) const {
        return std::sqrt(jsd(std::forward<Args>(args)...));
    }
    void set_lambda(FT param) {
        if(param < 0. || param > 1.)
            throw std::invalid_argument(std::string("Param for lambda ") + std::to_string(param) + " is out of range.");
        lambda_ = param;
    }
private:
    template<typename Container=blaze::DynamicVector<FT, blaze::rowVector>>
    void prep(Prior prior, const Container *c=nullptr) {
        row_sums_.reset(new VecT(data_.rows()));
        auto rowsumit = row_sums_->begin();
        for(auto r: blz::rowiterator(data_)) {
            CONST_IF(blz::IsDenseMatrix_v<MatrixType>) {
                if(prior == NONE) {
                r += 1e-50;
                assert(blz::min(r) > 0.);
                }
            }
            const auto countsum = blz::sum(r);
            r /= countsum;
            *rowsumit++ = countsum;
        }
        switch(prior) {
            case NONE:
            break;
            case DIRICHLET:
                CONST_IF(!IsSparseMatrix_v<MatrixType>) {
                    data_ += (1. / data_.columns());
                } else {
                    throw std::invalid_argument("Can't use Dirichlet prior for sparse matrix");
                }
                break;
            case GAMMA_BETA:
                if(c == nullptr) throw std::invalid_argument("Can't do gamma_beta with null pointer");
                CONST_IF(!IsSparseMatrix_v<MatrixType>) {
                    data_ += (1. / *std::begin(*c));
                } else {
                    throw std::invalid_argument("Can't use gamma beta prior for sparse matrix");
                }
            break;
            case FEATURE_SPECIFIC_PRIOR:
                if(c == nullptr) throw std::invalid_argument("Can't do feature-specific with null pointer");
                for(auto rw: blz::rowiterator(data_))
                    rw += *c;
        }
        for(size_t i = 0; i < data_.rows(); ++i)
            row(i) /= blaze::sum(row(i)); // Ensure that they sum to 1.

        if(detail::needs_logs(measure_)) {
            logdata_.reset(new MatrixType(neginf2zero(log(data_))));
        } else if(detail::needs_sqrt(measure_)) {
            sqrdata_.reset(new MatrixType(blz::sqrt(data_)));
        }
        llr_cache_.reset(new VecT(data_.rows()));
        jsdcache_.reset(new VecT(data_.rows()));
        for(size_t i = 0; i < llr_cache_->size(); ++i) {
            double cv = blaze::dot(row(i), logrow(i));
            llr_cache_->operator[](i) = cv * row_sums_->operator[](i);
            jsdcache_->operator[](i)  = cv;
        }
        VERBOSE_ONLY(std::cout << *logdata_;)
    }
}; // ProbDivApplicator

template<typename MT1, typename MT2>
struct PairProbDivApplicator {
    ProbDivApplicator<MT1> &pda_;
    ProbDivApplicator<MT2> &pdb_;
    PairProbDivApplicator(ProbDivApplicator<MT1> &lhs, ProbDivApplicator<MT2> &rhs): pda_(lhs), pdb_(rhs) {
        if(lhs.measure_ != rhs.measure_) throw std::runtime_error("measures must be the same (for preprocessing reasons).");
    }
    decltype(auto) operator()(size_t i, size_t j) const {
        return pda_(i, pdb_.row(j));
    }
};

template<typename MatrixType>
class MultinomialJSDApplicator: public ProbDivApplicator<MatrixType> {
    using super = ProbDivApplicator<MatrixType>;
    template<typename PriorContainer=blaze::DynamicVector<typename super::FT, blaze::rowVector>>
    MultinomialJSDApplicator(MatrixType &ref,
                             Prior prior=NONE,
                             const PriorContainer *c=nullptr):
        ProbDivApplicator<MatrixType>(ref, JSD, prior, c) {}
};
template<typename MatrixType>
class MultinomialLLRApplicator: public ProbDivApplicator<MatrixType> {
    using super = ProbDivApplicator<MatrixType>;
    template<typename PriorContainer=blaze::DynamicVector<typename super::FT, blaze::rowVector>>
    MultinomialLLRApplicator(MatrixType &ref,
                             Prior prior=NONE,
                             const PriorContainer *c=nullptr):
        ProbDivApplicator<MatrixType>(ref, LLR, prior, c) {}
};

template<typename MJD>
struct BaseOperand {
    using type = decltype(*std::declval<MJD>());
};

template<typename MatrixType, typename PriorContainer=blaze::DynamicVector<typename MatrixType::ElementType, blaze::rowVector>>
auto make_probdiv_applicator(MatrixType &data, ProbDivType type=JSM, Prior prior=NONE, const PriorContainer *pc=nullptr) {
    return ProbDivApplicator<MatrixType>(data, type, prior, pc);
}
template<typename MatrixType, typename PriorContainer=blaze::DynamicVector<typename MatrixType::ElementType, blaze::rowVector>>
auto make_jsm_applicator(MatrixType &data, Prior prior=NONE, const PriorContainer *pc=nullptr) {
    return ProbDivApplicator<MatrixType>(data, JSM, prior, pc);
}



template<typename MatrixType>
auto make_kmc2(const ProbDivApplicator<MatrixType> &app, unsigned k, size_t m=2000, uint64_t seed=13) {
    wy::WyRand<uint64_t> gen(seed);
    return coresets::kmc2(app, gen, app.size(), k, m);
}

template<typename MatrixType>
auto make_kmeanspp(const ProbDivApplicator<MatrixType> &app, unsigned k, uint64_t seed=13) {
    wy::WyRand<uint64_t> gen(seed);
    return coresets::kmeanspp(app, gen, app.size(), k);
}

template<typename MatrixType, typename WFT=typename MatrixType::ElementType, typename IT=uint32_t>
auto make_d2_coreset_sampler(const ProbDivApplicator<MatrixType> &app, unsigned k, uint64_t seed=13, const WFT *weights=nullptr, coresets::SensitivityMethod sens=cs::LBK) {
    wy::WyRand<uint64_t> gen(seed);
    auto [centers, asn, costs] = coresets::kmeanspp(app, gen, app.size(), k);
    //for(const auto c: centers) std::fprintf(stderr, "%u,", c);
    //std::fputc('\n', stderr);
    //std::fprintf(stderr, "costs size: %zu. centers size: %zu\n", costs.size(), centers.size());
    coresets::CoresetSampler<typename MatrixType::ElementType, IT> cs;
    cs.make_sampler(app.size(), centers.size(), costs.data(), asn.data(), weights,
                    /*seed=*/gen(), sens);
    return cs;
}

} // jsd
using jsd::ProbDivApplicator;
using jsd::make_d2_coreset_sampler;
using jsd::make_kmc2;
using jsd::make_kmeanspp;
using jsd::make_jsm_applicator;
using jsd::make_probdiv_applicator;



} // fgc

#endif
