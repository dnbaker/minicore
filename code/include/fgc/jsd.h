#ifndef FGC_JSD_H__
#define FGC_JSD_H__
#include "fgc/distance.h"
#include "distmat/distmat.h"
#include "fgc/kmeans.h"

namespace fgc {

namespace jsd {

using namespace blz;
using namespace distance;

template<typename MT, bool SO>
void fill_symmetric_upper_triangular(blaze::DenseMatrix<MT, SO> &mat) {
    diagonal(~mat) = 0.;
    const size_t nr = (~mat).rows();
    for(size_t i = 0; i < nr - 1; ++i) {
        submatrix(~mat, i + 1, i, nr - i - 1, 1) = trans(submatrix(~mat, i, i + 1, 1, nr - i - 1));
    }
}

template<typename MatrixType>
class MultinomialJSDApplicator {

    //using opposite_type = typename base_type::OppositeType;
    MatrixType &data_;
    using VecT = blaze::DynamicVector<typename MatrixType::ElementType>;
    std::unique_ptr<VecT> cached_cumulants_;
    std::unique_ptr<VecT> row_sums_;
    std::unique_ptr<MatrixType> logdata_;
public:
    using FT = typename MatrixType::ElementType;
    using MT = MatrixType;
    using This = MultinomialJSDApplicator<MatrixType>;
    using ConstThis = const MultinomialJSDApplicator<MatrixType>;

    const Prior prior_;
    const MatrixType &data() const {return data_;}
    size_t size() const {return data_.rows();}
    template<typename PriorContainer=blaze::DynamicVector<FT, blaze::rowVector>>
    MultinomialJSDApplicator(MatrixType &ref,
                             Prior prior=NONE,
                             const PriorContainer *c=nullptr):
        data_(ref), logdata_(nullptr), prior_(prior)
    {
        prep(c);
    }
    /*
     * Returns JSD
     */
    double operator()(size_t lhind, size_t rhind) const {
        return jsd(lhind, rhind);
    }
    /*
     * Sets distance matrix, under Jensen-Shannon metric (by default with use_jsm)
     * or Jensen-Shannon distance.
     */
    template<typename MatType>
    void set_distance_matrix(MatType &m, bool use_jsm=false) const {
        using blaze::sqrt;
        const size_t nr = m.rows();
        assert(nr == m.columns());
        assert(nr == data_.rows());
        for(size_t i = 0; i < nr; ++i) {
            OMP_PFOR
            for(size_t j = i + 1; j < nr; ++j) {
                auto v = jsd(i, j);
                m(i, j) = v;
            }
        }
        CONST_IF(blaze::IsDenseMatrix_v<MatType> || blaze::IsSparseMatrix_v<MatType>) {
            if(use_jsm) m = sqrt(m);
        }
#if 0
        } else {
            if(use_jsm) {
                blaze::CustomVector<std::decay_t<decltype(m(0, 0))>, blaze::unaligned, blaze::unpadded> cv(m.data(), m.size());
                cv = sqrt(cv);
            }
        }
#endif
        //std::fprintf(stderr, "Note: matrix representation only sets upper triangular entries for space considerations");
    }
    template<typename MatType>
    void set_llr_matrix(MatType &m) const {
        using blaze::sqrt;
        const size_t nr = m.rows();
        assert(nr == m.columns());
        assert(nr == data_.rows());
        for(size_t i = 0; i < nr; ++i) {
            OMP_PFOR
            for(size_t j = i + 1; j < nr; ++j)
                m(i, j) = llr(i, j);
        }
        CONST_IF(blaze::IsDenseMatrix_v<MatType>) {
            //fill_symmetric_upper_triangular(m);
        } else {
            //std::fprintf(stderr, "Note: sparse matrix representation only sets upper triangular entries for space considerations");
        }
    }
    blaze::DynamicMatrix<float> make_distance_matrix(bool use_jsm=true) const {
        std::fprintf(stderr, "About to make distance matrix of %zu/%zu with %s calculated\n", data_.rows(), data_.rows(), use_jsm ? "jsm": "jsd");
        blaze::DynamicMatrix<float> ret(data_.rows(), data_.rows());
        set_distance_matrix(ret, use_jsm);
        return ret;
    }
    double jsd(size_t lhind, size_t rhind) const {
        assert(lhind < data_.rows());
        assert(rhind < data_.rows());
        double ret;
        if(likely(logdata_)) {
            ret =  bnj::multinomial_jsd(row(lhind),
                                        row(rhind), logrow(lhind), logrow(rhind));
        } else {
            ret =  bnj::multinomial_jsd(row(lhind),
                                        row(rhind),
                                        map(blz::log(row(lhind)), NegInf2Zero()),
                                        map(blz::log(row(rhind)), NegInf2Zero()));
        }
        static constexpr typename MatrixType::ElementType threshold
            = std::is_same_v<typename MatrixType::ElementType, double>
                                ? 0.: -1e-5;
        assert(ret >= threshold || !std::fprintf(stderr, "ret: %g\n", ret));
        return std::max(ret, 0.);
    }
    auto weighted_row(size_t ind) const {
        return blz::row(data_, ind BLAZE_CHECK_DEBUG) * row_sums_->operator[](ind);
    }
    auto row(size_t ind) const {return blz::row(data_, ind BLAZE_CHECK_DEBUG);}
    auto logrow(size_t ind) const {return blz::row(*logdata_, ind BLAZE_CHECK_DEBUG);}
    double llr(size_t lhind, size_t rhind) const {
        double ret =
            // X_j^Tlog(p_j)
            blaze::dot(row(lhind), logrow(lhind)) * row_sums_->operator[](lhind)
            +
            // X_k^Tlog(p_k)
            blaze::dot(row(rhind), logrow(rhind)) * row_sums_->operator[](rhind)
            -
            // (X_k + X_j)^Tlog(p_jk)
            blaze::dot(weighted_row(lhind) + weighted_row(rhind),
                blz::map(blz::log(0.5 * (row(lhind) + row(rhind))),
                         NegInf2Zero())
            );
        assert(ret >= -1e-4 * (row_sums_->operator[](lhind) + row_sums_->operator[](rhind)) || !std::fprintf(stderr, "ret: %g\n", ret));
        return std::max(ret, 0.);
    }
    double jsm(size_t lhind, size_t rhind) const {
        return std::sqrt(jsd(lhind, rhind));
    }
private:
    template<typename Container=blaze::DynamicVector<FT, blaze::rowVector>>
    void prep(const Container *c=nullptr) {
        row_sums_.reset(new VecT(data_.rows()));
        auto rowsumit = row_sums_->begin();
        for(auto r: blz::rowiterator(data_)) {
            const auto countsum = blz::sum(r);
            r /= countsum;
            *rowsumit++ = countsum;
        }
        //blz::normalize(data_);
        switch(prior_) {
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
            row(i) /= blaze::l2Norm(row(i));
        logdata_.reset(new MatrixType(data_.rows(), data_.columns()));
        *logdata_ = blaze::map(blaze::log(data_), NegInf2Zero());
        VERBOSE_ONLY(std::cout << *logdata_;)
    }
}; // MultinomialJSDApplicator

template<typename MJD>
struct BaseOperand {
    using type = decltype(*std::declval<MJD>());
};

template<typename MatrixType, typename PriorContainer=blaze::DynamicVector<typename MatrixType::ElementType, blaze::rowVector>>
auto make_jsd_applicator(MatrixType &data, Prior prior=NONE, const PriorContainer *pc=nullptr) {
    return MultinomialJSDApplicator<MatrixType>(data, prior, pc);
}



template<typename MatrixType>
auto make_kmc2(const jsd::MultinomialJSDApplicator<MatrixType> &app, unsigned k, size_t m=2000, uint64_t seed=13) {
    wy::WyRand<uint64_t> gen(seed);
    return coresets::kmc2(app, gen, app.size(), k, m);
}

template<typename MatrixType>
auto make_kmeanspp(const jsd::MultinomialJSDApplicator<MatrixType> &app, unsigned k, uint64_t seed=13) {
    wy::WyRand<uint64_t> gen(seed);
    return coresets::kmeanspp(app, gen, app.size(), k);
}

} // jsd


} // fgc

#endif
