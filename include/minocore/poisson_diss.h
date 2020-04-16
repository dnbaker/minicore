#ifndef FGC_PSD_H__
#define FGC_PSD_H__
#include "minocore/distance.h"
#include "distmat/distmat.h"
#include "minocore/kmeans.h"
#include "minocore/coreset.h"
#include "minocore/Inf2Zero.h"

namespace minocore {

namespace pd {
using namespace blz;

template<typename Matrix>
class PoissonDissimilarityApplicator {
    using ET = typename Matrix::ElementType;
    using MatrixType = std::decay_t<decltype(~std::declval<Matrix>())>;
    Matrix mat_;
    blz::DV<ET, blz::rowVector> column_sums_;
    blz::DV<ET, blz::columnVector> row_sums_;
    blz::DV<ET>                    contribs_;
    ET total_sum_;
public:
    template<typename Container=const blz::DV<ET>>
    PoissonDissimilarityApplicator(Matrix &mat, const Container *c=nullptr, Prior prior=NONE): mat_(mat) {
        auto rowsumit = row_sums_.begin();
        for(auto r: blz::rowiterator(mat_)) {
            CONST_IF(blz::IsDenseMatrix_v<MatrixType>) {
                if(prior == Prior::NONE) {
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
                    mat_ += (1. / mat_.columns());
                } else {
                    throw std::invalid_argument("Can't use Dirichlet prior for sparse matrix");
                }
                break;
            case GAMMA_BETA:
                if(c == nullptr) throw std::invalid_argument("Can't do gamma_beta with null pointer");
                CONST_IF(!IsSparseMatrix_v<MatrixType>) {
                    mat_ += (1. / *std::begin(*c));
                } else {
                    throw std::invalid_argument("Can't use gamma beta prior for sparse matrix");
                }
            break;
            case FEATURE_SPECIFIC_PRIOR:
                if(c == nullptr) throw std::invalid_argument("Can't do feature-specific with null pointer");
                for(auto rw: blz::rowiterator(mat_))
                    rw += *c;
        }
        column_sums_ = blz::sum<columnwise>(mat_);
        row_sums_    = blz::sum<rowwise>(mat_);
        total_sum_ = blz::sum(row_sums_);
        for(size_t i = 0; i < mat_.rows(); ++i)
            row(i) /= blaze::sum(row(i)); // Ensure that they sum to 1.
        for(size_t i = 0; i < mat_.rows(); ++i) {
            auto r = row(i);
            contribs_[i] = row_sums_[i] * blz::sum(r)
                          - row_sums_[i] * blz::dot(r, r)
                          + row_sums_[i] * blz::dot(r, neginf2zero(blz::log(r)));
        }
        //auto make_n_ij = [&](size_t i, size_t j) {return column_sums_[j] * row_sums_[i] / total_sum_;};
    }
    auto row(size_t i) {
        return row(i, mat_ BLAZE_CHECK_DEBUG);
    }
    auto row(size_t i) const {
        return row(i, mat_ BLAZE_CHECK_DEBUG);
    }
    double llr(size_t i, size_t j) const {
        return contribs_[i] + contribs_[j];
    }
};

}

}
#endif /* FGC_PSD_H__ */
