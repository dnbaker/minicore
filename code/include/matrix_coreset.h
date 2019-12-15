#pragma once
#include "coreset.h"
#include "blaze_adaptor.h"

namespace coresets {

template<typename MatrixType, typename FT=double>
struct MatrixCoreset {
    MatrixType mat_;
    std::vector<FT> weights_;
    bool rowwise_;
    MatrixCoreset &merge(const MatrixCoreset &o) {
        if(rowwise_ != o.rowwise_) throw std::runtime_error("Can't merge coresets of differing rowwiseness");
        weights_.reserve(weights_.size() + o.weights_.size());
        weights_.insert(weights_.end(), o.weights_.begin(), o.weights_.end());
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

template<typename FT, typename IT, typename MatrixType>
auto index2matrix(const IndexCoreset<IT, FT> &ic, const MatrixType &mat, bool rowwise) {
    auto weights = ic.weights_;
    MatrixType ret;
    if(rowwise) {
        auto rows = rows(mat, ic.indices_);
        ret.resize(rows.rows(), rows.columns());
        ret = rows;
    } else {
        auto columns = columns(mat, ic.indices_);
        ret.resize(columns.rows(), columns.columns());
        ret = columns;
    }
    return MatrixCoreset<MatrixType, FT>{std::move(ret), std:move(weights), rowwise};
} // index2matrix

} // coresets
