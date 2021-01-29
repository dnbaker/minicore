#pragma once
#include "minicore/coreset/coreset.h"

namespace minicore {
namespace coresets {

template<typename MatrixType, typename FT=double>
struct MatrixCoreset {
    MatrixType mat_;
    blaze::DynamicVector<FT> weights_;
    bool rowwise_;
    MatrixCoreset &merge(const MatrixCoreset &o) {
        if(rowwise_ != o.rowwise_) throw std::runtime_error("Can't merge coresets of differing rowwiseness");
        weights_.reserve(weights_.size() + o.weights_.size());
        //weights_.insert(weights_.end(), o.weights_.begin(), o.weights_.end());
        for(auto w: o.weights_) push_back(weights_, w);
        if(rowwise_) {
            assert(mat_.columns() == o.mat_.columns());
            auto nc = mat_.columns();
            size_t oldr = mat_.rows();
            mat_.resize(mat_.rows() + o.mat_.rows(), mat_.columns());
            submatrix(mat_, oldr, 0, o.mat_.rows(), nc) = o.mat_;
        } else {
            assert(mat_.rows() == o.mat_.rows());
            auto nr = mat_.rows();
            size_t oldc = mat_.columns();
            mat_.resize(nr, mat_.columns() + o.mat_.columns());
            submatrix(mat_, 0, oldc, nr, o.mat_.columns()) = o.mat_;
        }
        return *this;
    }
    MatrixCoreset &operator+=(const MatrixType &o) {return this->merge(o);}
    MatrixCoreset operator+(const MatrixType &o) const {
        MatrixCoreset ret(*this);
        ret += o;
        return ret;
    }
};

template<typename Mat, typename View>
void resize_and_assign(Mat &dest, const View &view) {
    dest.resize(view.rows(), view.columns());
    dest = view;
}

template<typename FT, typename IT, typename MatrixType, typename CMatrixType=blaze::DynamicMatrix<FT>>
MatrixCoreset<MatrixType, FT>
index2matrix(const IndexCoreset<IT, FT> &ic, const MatrixType &mat,
             bool rowwise=(blaze::StorageOrder_v<MatrixType> == blaze::rowMajor))
{
    auto weights = ic.weights_;
    CMatrixType ret;
    const auto icdat = ic.indices_.data();
    const size_t icsz = ic.indices_.size();
    if(rowwise) {
#if !NDEBUG
        for(size_t i = 0; i < icsz; ++i) assert(icdat[i] < mat.rows());
#endif
        auto rows = blaze::rows(mat, icdat, icsz);
        resize_and_assign(ret, rows);
    } else {
#if !NDEBUG
        for(size_t i = 0; i < icsz; ++i) assert(icdat[icsz] < mat.columns());
#endif
        auto columns = blaze::columns(mat, icdat, icsz);
        resize_and_assign(ret, columns);
    }
    return MatrixCoreset<MatrixType, FT>{std::move(ret), std::move(weights), rowwise};
} // index2matrix
} // coresets

} // minicore
