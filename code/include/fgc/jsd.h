#ifndef FGC_JSD_H__
#define FGC_JSD_H__
#include "fgc/distance.h"
#include "distmat/distmat.h"
#include "fgc/kmeans.h"

namespace fgc {

namespace distance {

using namespace blz;
using namespace distance;


enum ProbDivType {
    L1  = 0,
    L2  = 1,
    SQRL2  = 2,
    JSM = 4, // Multinomial Jensen-Shannon Metric
    JSD = 5, // Multinomial Jensen-Shannon Divergence
    MKL = 6, // Multinomial KL Divergence
    POISSON = 7, // Poisson KL
    PSD = JSD, // Poisson JSD, but algebraically equivalent
    PSM = JSM,
    HELLINGER = 8,
    BHATTACHARYA_METRIC = 9,
    BHATTACHARYA_DISTANCE = 10,
    LLR = 12,
};

namespace detail {
INLINE bool  needs_logs(ProbDivType d) {
    return d == JSM || d == JSD || d == MKL || d == POISSON || d == LLR;
}
INLINE bool  needs_sqrt(ProbDivType d) {
    return d == HELLINGER || d == BHATTACHARYA_METRIC || d == BHATTACHARYA_DISTANCE;
}
const char *prob2str(ProbDivType d) {
    switch(d) {
        case L1: return "L1";
        case L2: return "L2";
        case SQRL2: return "SQRL2";
        case JSD: return "JSD";
        case JSM: return "JSM";
        case MKL: return "MKL";
        case POISSON: return "POISSON";
        case HELLINGER: return "HELLINGER";
        case LLR: return "LLR";
        case BHATTACHARYA_DISTANCE: return "BHATTACHARYA_DISTANCE";
        case BHATTACHARYA_METRIC: return "BHATTACHARYA_METRIC";
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
     * Returns divergence/measure
     */
    INLINE double operator()(size_t i, size_t j) const {
        return this->operator()(i, j, measure_);
    }
    INLINE double operator()(size_t i, size_t j, ProbDivType measure) const {
        double ret;
        switch(measure) {
            case L1: ret = l1Norm(row(i) - row(j)); break;
            case L2: ret = l2Norm(row(i) - row(j)); break;
            case SQRL2: ret = blaze::sqrNorm(row(i) - row(j)); break;
            case JSD: ret = jsd(i, j); break;
            case JSM: ret = jsm(i, j); break;
            case MKL: ret = mkl(i, j); break;
            case POISSON: ret = pkl(i, j); break;
            case HELLINGER: ret = hellinger(i, j); break;
            case BHATTACHARYA_METRIC: ret = bhattacharya_metric(i, j); break;
            case BHATTACHARYA_DISTANCE: ret = bhattacharya_distance(i, j); break;
            case LLR: return llr(i, j); break;
            default: __builtin_unreachable();
        }
        return ret;
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
        for(size_t i = 0; i < nr; ++i) {
            OMP_PFOR
            for(size_t j = i + 1; j < nr; ++j) {
                auto v = this->operator()(i, j, measure);
                m(i, j) = v;
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
        if(symmetrize) {
            fill_symmetric_upper_triangular(m);
        }
    }
    blaze::DynamicMatrix<float> make_distance_matrix() const {
        return make_distance_matrix(measure_);
    }
    blaze::DynamicMatrix<float> make_distance_matrix(ProbDivType measure, bool symmetrize=false) const {
        std::fprintf(stderr, "About to make distance matrix of %zu/%zu with %s calculated\n", data_.rows(), data_.rows(), detail::prob2str(measure));
        blaze::DynamicMatrix<float> ret(data_.rows(), data_.rows());
        set_distance_matrix(ret, measure, symmetrize);
        return ret;
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
        if(likely(logdata_)) {
            ret =  bnj::multinomial_jsd(ri,
                                        rj, logrow(i), logrow(j));
        } else {
            ret =  bnj::multinomial_jsd(ri,
                                        rj,
                                        map(blz::log(ri), NegInf2Zero()),
                                        map(blz::log(rj), NegInf2Zero()));
        }
#ifndef NDEBUG
        static constexpr typename MatrixType::ElementType threshold
            = std::is_same_v<typename MatrixType::ElementType, double>
                                ? 0.: -1e-5;
        assert(ret >= threshold || !std::fprintf(stderr, "ret: %g (numerical stability issues)\n", ret));
#endif
        return std::max(ret, 0.);
    }
    auto weighted_row(size_t ind) const {
        return blz::row(data_, ind BLAZE_CHECK_DEBUG) * row_sums_->operator[](ind);
    }
    auto row(size_t ind) const {return blz::row(data_, ind BLAZE_CHECK_DEBUG);}
    auto logrow(size_t ind) const {return blz::row(*logdata_, ind BLAZE_CHECK_DEBUG);}
    auto sqrtrow(size_t ind) const {return blz::row(*sqrdata_, ind BLAZE_CHECK_DEBUG);}
    double mkl(size_t i, size_t j) const {
        // Multinomial KL
        return blz::dot(row(i), logrow(i) - logrow(j));
    }
    double pkl(size_t i, size_t j) const {
        // Poission KL
        return blz::dot(row(i), logrow(i) - logrow(j)) + blz::sum(row(j) - row(i));
    }
    double psd(size_t i, size_t j) const {
        // Poission JSD
        auto logmn = evaluate(log(.5 * (row(i) + row(j))));
        return .5 * (blz::dot(row(i), logrow(i) - logmn) + blz::dot(row(j), logrow(j) - logmn));
    }
    double bhattacharya_sim(size_t i, size_t j) const {
        return sqrdata_ ? blz::dot(sqrtrow(i), sqrtrow(j))
                        : blz::sum(blz::sqrt(row(i) * row(j)));
    }
    double bhattacharya_distance(size_t i, size_t j) const {
        return -std::log(bhattacharya_sim(i, j));
    }
    double bhattacharya_metric(size_t i, size_t j) const {
        return std::sqrt(1 - bhattacharya_sim(i, j));
    }
    double psm(size_t i, size_t j) const {return std::sqrt(psd(i, j));}
    double llr(size_t i, size_t j) const {
        double ret =
            // X_j^Tlog(p_j)
            blaze::dot(row(i), logrow(i)) * row_sums_->operator[](i)
            +
            // X_k^Tlog(p_k)
            blaze::dot(row(j), logrow(j)) * row_sums_->operator[](j)
            -
            // (X_k + X_j)^Tlog(p_jk)
            blaze::dot(weighted_row(i) + weighted_row(j),
                blz::map(blz::log(0.5 * (row(i) + row(j))),
                         NegInf2Zero())
            );
        assert(ret >= -1e-3 * (row_sums_->operator[](i) + row_sums_->operator[](j)) || !std::fprintf(stderr, "ret: %g\n", ret));
        return std::max(ret, 0.);
    }
    double jsm(size_t i, size_t j) const {
        return std::sqrt(jsd(i, j));
    }
private:
    template<typename Container=blaze::DynamicVector<FT, blaze::rowVector>>
    void prep(Prior prior, const Container *c=nullptr) {
        row_sums_.reset(new VecT(data_.rows()));
        auto rowsumit = row_sums_->begin();
        for(auto r: blz::rowiterator(data_)) {
            const auto countsum = blz::sum(r);
            r /= countsum;
            *rowsumit++ = countsum;
        }
        //blz::normalize(data_);
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
            logdata_.reset(new MatrixType(data_.rows(), data_.columns()));
            *logdata_ = blaze::map(blaze::log(data_), NegInf2Zero());
        } else if(detail::needs_sqrt(measure_)) {
            sqrdata_.reset(new MatrixType(blz::sqrt(data_)));
        }
        VERBOSE_ONLY(std::cout << *logdata_;)
    }
}; // ProbDivApplicator

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

} // jsd


} // fgc

#endif
