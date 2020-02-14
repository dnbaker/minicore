#ifndef FGC_JSD_H__
#define FGC_JSD_H__
#include "fgc/distance.h"
#include "distmat/distmat.h"

namespace fgc {

namespace jsd {

using namespace blz;
using namespace distance;

template<typename MatrixType>
class MultinomialJSDApplicator {
    using FT = typename MatrixType::ElementType;

    //using opposite_type = typename base_type::OppositeType;
    MatrixType &data_;
    std::unique_ptr<blaze::DynamicVector<FT>> cached_cumulants_;
    std::unique_ptr<MatrixType> logdata_;
public:
    const Prior prior_;
    template<typename PriorContainer=blaze::DynamicVector<FT, blaze::rowVector>>
    MultinomialJSDApplicator(MatrixType &ref,
                             Prior prior=NONE,
                             const PriorContainer *c=nullptr,
                             bool use_mj_kl=true):
        data_(ref), logdata_(nullptr), prior_(prior)
    {
        prep(c, use_mj_kl);
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
    void set_distance_matrix(MatType &m, bool use_jsm=true) const {
        using blaze::sqrt;
        const size_t nr = m.rows();
        assert(nr == m.columns());
        assert(nr == data_.rows());
        for(size_t i = 0; i < nr; ++i) {
            OMP_PFOR
            for(size_t j = i + 1; j < nr; ++j) {
                auto v = jsd(i, j);
                if(!blaze::IsDenseMatrix_v<MatrixType> && use_jsm) v = std::sqrt(v);
                m(i, j) = v;
            }
        }
        CONST_IF(blaze::IsDenseMatrix_v<MatrixType>) {
            diagonal(m) = 0.;
            for(size_t i = 0; i < nr - 1; ++i) {
                submatrix(m, i + 1, i, nr - i - 1, 1) = trans(submatrix(m, i, i + 1, 1, nr - i - 1));
            }
            if(use_jsm)
                m = sqrt(m);
        } else {
            std::fprintf(stderr, "Note: sparse matrix representation only sets upper triangular entries for space considerations");
        }
    }
    blaze::DynamicMatrix<float> make_distance_matrix(bool use_jsm=true) const {
        std::fprintf(stderr, "About to make distance matrix of %zu/%zu with %s calculated\n", data_.rows(), data_.rows(), use_jsm ? "jsm": "jsd");
        blaze::DynamicMatrix<float> ret(data_.rows(), data_.rows());
        set_distance_matrix(ret, use_jsm);
        return ret;
    }
    double jsd_bnj(size_t lhind, size_t rhind) const {
#if VERBOSE_AF
        std::fprintf(stderr, "Banerjee\n");
#endif
        double ret;
        if(logdata_) {
            ret =  bnj::multinomial_jsd(row(data_, lhind),
                                        row(data_, rhind),
                                        row(*logdata_, lhind),
                                        row(*logdata_, rhind));
        } else {
            ret =  bnj::multinomial_jsd(row(data_, lhind),
                                        row(data_, rhind),
                                        map(map(row(data_, lhind), blz::Log()), NegInf2Zero()),
                                        map(map(row(data_, rhind), blz::Log()), NegInf2Zero()));
        }
        return ret;
    }
    double jsd_mj(size_t lhind, size_t rhind) const {
        assert(cached_cumulants_.get());
        assert(lhind < cached_cumulants_->size());
        assert(rhind < cached_cumulants_->size());
        const auto lhv = cached_cumulants_->operator[](lhind),
                   rhv = cached_cumulants_->operator[](rhind);
        double ret;
        if(logdata_) {
            assert(logdata_->rows() == data_.rows());
            ret = mj::multinomial_jsd(row(data_, lhind),
                                      row(data_, rhind),
                                      row(*logdata_, lhind),
                                      row(*logdata_, rhind),
                                      lhv,
                                      rhv);
        } else {
            ret = mj::multinomial_jsd(row(data_, lhind),
                                      row(data_, rhind),
                                      lhv,
                                      rhv);
        }
        return ret;
    }
    double jsd(size_t lhind, size_t rhind) const {
        assert(lhind < data_.rows());
        assert(rhind < data_.rows());
        return cached_cumulants_ ? jsd_mj(lhind, rhind): jsd_bnj(lhind, rhind);
    }
    double jsm(size_t lhind, size_t rhind) const {
        return std::sqrt(jsd(lhind, rhind));
    }
private:
    template<typename Container=blaze::DynamicVector<FT, blaze::rowVector>>
    void prep(const Container *c=nullptr, bool use_mj_kl=true) {
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
            row(data_, i) /= blaze::l2Norm(row(data_, i));
        logdata_.reset(new MatrixType(data_.rows(), data_.columns()));
        *logdata_ = blaze::map(blaze::log(data_), NegInf2Zero());
        VERBOSE_ONLY(std::cout << *logdata_;)
        if(use_mj_kl) {
            std::fprintf(stderr, "Making cached cumulants. num rows: %zu\n", data_.rows());
            cached_cumulants_.reset(new blz::DV<FT>(data_.rows()));
            for(size_t i = 0; i < data_.rows(); ++i)
                cached_cumulants_->operator[](i) = multinomial_cumulant(row(data_, i));
            std::fprintf(stderr, "Made cached cumulants. num rows: %zu\n", data_.rows());
        }
    }
}; // MultinomialJSDApplicator

template<typename MatrixType, typename PriorContainer=blaze::DynamicVector<typename MatrixType::ElementType, blaze::rowVector>>
auto make_jsd_applicator(MatrixType &data, Prior prior=NONE, const PriorContainer *pc=nullptr, bool use_mj_kl=true) {
    return MultinomialJSDApplicator<MatrixType>(data, prior, pc, use_mj_kl);
}

} // jsd


} // fgc

#endif
