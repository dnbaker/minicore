#pragma once
#if defined(__has_include) && __has_include("sleef.h")
extern "C" {
#  include "sleef.h"
}
#endif
#include "aesctr/wy.h"
#include "blaze/Math.h"
#include <distmat/distmat.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "./shared.h"
#include "./Inf2Zero.h"

namespace blz {
using blaze::unchecked;

// These blaze adaptors exist for the purpose of
// providing a pair of iterators.
template<typename this_type>
struct row_iterator_t {
    size_t rownum;
    this_type &ref_;
    row_iterator_t(size_t rn, this_type &ref): rownum(rn), ref_(ref) {}

    auto index() const {return rownum;}
    INLINE row_iterator_t &operator++() {++rownum; return *this;}
    INLINE row_iterator_t &operator--() {--rownum; return *this;}
    INLINE row_iterator_t operator++(int) {
        row_iterator_t ret{rownum, ref_};
        ++rownum;
        return ret;
    }
    INLINE row_iterator_t operator--(int) {
        row_iterator_t ret{rownum, ref_};
        --rownum;
        return ret;
    }
    bool operator==(row_iterator_t o) const {return o.rownum == rownum;}
    bool operator!=(row_iterator_t o) const {return o.rownum != rownum;}
    bool operator<(row_iterator_t o) const {
        return rownum < o.rownum;
    }
    bool operator<=(row_iterator_t o) const {return rownum <= o.rownum;}
    bool operator>(row_iterator_t o) const {return rownum > o.rownum;}
    bool operator>=(row_iterator_t o) const {return rownum >= o.rownum;}
    std::ptrdiff_t operator-(row_iterator_t o) const {return rownum - o.rownum;}
    auto operator[](size_t index) const {
        //std::fprintf(stderr, "index: %zu. rownum: %zu. nrows: %zu\n", index, rownum, ref_.rows());
        assert(index + rownum < ref_.rows());
        return row(ref_, index + rownum, blaze::unchecked);
    }
    auto operator*() const {
        assert(rownum < ref_.rows());
        return row(ref_, rownum, blaze::unchecked);
    }
};

template<typename this_type>
struct column_iterator_t {
    size_t columnnum;
    this_type &ref_;

    column_iterator_t(size_t rn, this_type &ref): columnnum(rn), ref_(ref) {}
    auto index() const {return columnnum;}
    column_iterator_t &operator++() {++columnnum; return *this;}
    column_iterator_t &operator--() {--columnnum; return *this;}
    column_iterator_t operator++(int) {
        column_iterator_t ret{columnnum, ref_};
        ++columnnum;
        return ret;
    }
    column_iterator_t operator--(int) {
        column_iterator_t ret{columnnum, ref_};
        --columnnum;
        return ret;
    }
    bool operator==(column_iterator_t o) const {return o.columnnum == columnnum;}
    bool operator!=(column_iterator_t o) const {return o.columnnum != columnnum;}
    bool operator<(column_iterator_t o) const {return columnnum < o.columnnum;}
    bool operator<=(column_iterator_t o) const {return columnnum <= o.columnnum;}
    bool operator>(column_iterator_t o) const {return columnnum > o.columnnum;}
    bool operator>=(column_iterator_t o) const {return columnnum >= o.columnnum;}
    std::ptrdiff_t operator-(column_iterator_t o) const {return columnnum - o.columnnum;}
    auto operator[](size_t index) const {
        assert(index + columnnum < ref_.columns());
        return column(ref_, index + columnnum, blaze::unchecked);
    }
    auto operator*() const {
        assert(columnnum < ref_.columns());
        return column(ref_, columnnum, blaze::unchecked);
    }
};

template<typename MatType>
struct RowViewer {
    const row_iterator_t<MatType> start_, end_;
    RowViewer(MatType &mat): start_(0, mat), end_(mat.rows(), mat) {}
    auto begin() const {return start_;}
    auto end()   const {return end_;}
};


template<typename MatType>
struct ColumnViewer {
    const column_iterator_t<MatType> start_, end_;
    ColumnViewer(MatType &mat): start_(0, mat), end_(mat.columns(), mat) {}
    auto begin() const {return start_;}
    auto end()   const {return end_;}
};
template<typename MatType>
struct ConstRowViewer: public RowViewer<const MatType> {
    ConstRowViewer(const MatType &mat): RowViewer<const MatType>(mat) {}
};
template<typename MatType>
struct ConstColumnViewer: public ColumnViewer<const MatType> {
    ConstColumnViewer(const MatType &mat): ColumnViewer<const MatType>(mat) {}
};

#define DOFUNC(fn) auto fn() const {return (~*this).fn();}
#define ADD_FUNCS\
    DOFUNC(rows)\
    DOFUNC(spacing)\
    /*DOFUNC(size)*/\
    DOFUNC(capacity)\
    DOFUNC(isNan)\
    DOFUNC(isSquare)\
    DOFUNC(isSymmetric)\
    DOFUNC(isLower)\
    DOFUNC(isUnilower)\
    DOFUNC(columns)

template<typename FT, bool SO=blaze::rowMajor>
struct DynamicMatrix: public blaze::DynamicMatrix<FT, SO> {
    using super = blaze::DynamicMatrix<FT, SO>;
    using this_type = DynamicMatrix<FT, SO>;
    template<typename...Args>
    DynamicMatrix<FT, SO>(Args &&...args): super(std::forward<Args>(args)...) {}
    struct row_iterator: public row_iterator_t<this_type> {};
    struct const_row_iterator: public row_iterator_t<const this_type> {};
    struct column_iterator: public column_iterator_t<this_type> {};
    struct const_column_iterator: public column_iterator_t<const this_type> {};
    template<typename...Args> this_type &operator=(Args &&...args) {
        ((super &)*this).operator=(std::forward<Args>(args)...);
        return *this;
    }
    auto rowiterator()       {return RowViewer<this_type>(*this);}
    auto rowiterator() const {return ConstRowViewer<this_type>(*this);}
    auto columniterator()       {return ColumnViewer<this_type>(*this);}
    auto columniterator() const {return ConstColumnViewer<this_type>(*this);}
    ADD_FUNCS
};

template< typename Type, blaze::AlignmentFlag AF, blaze::PaddingFlag PF, bool SO >
struct CustomMatrix: public blaze::CustomMatrix<Type, AF, PF, SO> {
    using super = blaze::CustomMatrix<Type, AF, PF, SO>;
    using this_type = CustomMatrix<Type, AF, PF, SO>;
    template<typename...Args>
    CustomMatrix(Args &&...args): super(std::forward<Args>(args)...) {}
    struct row_iterator: public row_iterator_t<this_type> {};
    struct const_row_iterator: public row_iterator_t<const this_type> {};
    struct column_iterator: public column_iterator_t<this_type> {};
    struct const_column_iterator: public column_iterator_t<const this_type> {};
    template<typename...Args> this_type &operator=(Args &&...args) {
        ((super &)*this).operator=(std::forward<Args>(args)...);
        return *this;
    }
    auto rowiterator()       {return RowViewer<this_type>(*this);}
    auto rowiterator() const {return ConstRowViewer<this_type>(*this);}
    auto columniterator()       {return ColumnViewer<this_type>(*this);}
    auto columniterator() const {return ConstColumnViewer<this_type>(*this);}
    ADD_FUNCS
};
#undef ADD_FUNCS
#undef DOFUNC


template<typename FT, bool SO=blaze::rowMajor>
using DM = DynamicMatrix<FT, SO>;
template<typename FT, bool TF=blaze::columnVector>
using DV = blaze::DynamicVector<FT, TF>;
template<typename FT, blaze::AlignmentFlag AF=blaze::unaligned, blaze::PaddingFlag PF=blaze::unpadded, bool SO=blaze::rowMajor>
using CM = CustomMatrix<FT, AF, PF, SO>;
template<typename FT, blaze::AlignmentFlag AF=blaze::unaligned, blaze::PaddingFlag PF=blaze::unpadded, bool TF=blaze::columnVector>
using CV = blaze::CustomVector<FT, AF, PF, TF>;
template<typename FT, bool SO=blaze::rowMajor>
using SM = blaze::CompressedMatrix<FT, SO>;
template<typename FT, bool TF=blaze::columnVector>
using SV = blaze::CompressedVector<FT, TF>;


template<typename MatType>
auto rowiterator(MatType &mat) {
    return RowViewer<MatType>(mat);
}
template<typename MatType>
auto rowiterator(const MatType &mat) {
    return ConstRowViewer<MatType>(mat);
}
template<typename MatType>
auto constrowiterator(const MatType &mat) {
    return ConstRowViewer<const MatType>(mat);
}
template<typename MatType>
auto columniterator(MatType &mat) {
    return ColumnViewer<MatType>(mat);
}
template<typename MatType>
auto columniterator(const MatType &mat) {
    return ConstColumnViewer<MatType>(mat);
}
template<typename MatType>
auto constcolumniterator(const MatType &mat) {
    return ConstColumnViewer<const MatType>(mat);
}


template<typename F, typename IT=std::uint32_t, size_t N=16>
auto indices_if(const F &func, size_t n) {
    blaze::SmallArray<IT, N> ret;
    for(IT i = 0; i < n; ++i)
        if(func(i)) ret.pushBack(i);
    return ret;
}

template<typename M, typename F, typename IT=std::uint32_t>
auto rows_if(const M &mat, const F &func) {
    auto wrapfunc = [&](auto x) -> bool {return func(row(mat, x, blaze::unchecked));};
    return rows(mat,
                indices_if<decltype(wrapfunc),IT>(wrapfunc, // predicate
                                                  mat.rows())); // nrow
}

template<typename M, typename F, typename IT=std::uint32_t>
auto columns_if(const M &mat, const F &func) {
    auto wrapfunc = [&](auto x) -> bool {return func(column(mat, x, blaze::unchecked));};
    return columns(mat, // matrix
                   indices_if<decltype(wrapfunc), IT>(wrapfunc, // predicate
                                                      mat.columns())); // ncol
}

template<typename VT, typename Allocator, size_t N, typename VT2>
INLINE auto push_back(blaze::SmallArray<VT, N, Allocator> &x, VT2 v) {
    return x.pushBack(v);
}

template<typename VT, typename Allocator, typename VT2>
INLINE auto push_back(std::vector<VT, Allocator> &x, VT2 v) {
    return x.push_back(v);
}
template<typename MatType>
static INLINE
void _assert_all_nonzero_(const MatType &x, const char *funcname, const char *filename, int linenum) {
    const auto nnz = ::blaze::nonZeros(x);
    if(unlikely(nnz != 0)) {
        std::fprintf(stderr, "[%s:%s:%d] assert all_nonzero failed: %zu\n", funcname, filename, linenum, size_t(nnz));
        std::abort();
    }
}

template<typename Item>
INLINE decltype(auto) sum(const Item &vec) {return blaze::sum(vec);}

template<typename FT, typename Alloc>
INLINE auto sum(const std::vector<FT, Alloc> &vec) {
    return blaze::sum(blaze::CustomVector<FT, blaze::unaligned, blaze::unpadded>(const_cast<FT *>(vec.data()), vec.size()));
}

template<typename MT, bool SO>
void fill_helper(blaze::Matrix<MT, SO> &mat) {
    diagonal(~mat) = 0.;
    const size_t nr = (~mat).rows();
    for(size_t i = 0; i < nr - 1; ++i) {
        submatrix(~mat, i + 1, i, nr - i - 1, 1) = trans(submatrix(~mat, i, i + 1, 1, nr - i - 1));
    }
}

template<typename FT >
void fill_helper(dm::DistanceMatrix<FT> &) {
     std::fprintf(stderr, "[%s] Warning: trying to fill_symmetric_upper_triangular on an unsupported type. Doing nothing.\n", __PRETTY_FUNCTION__);
}

template<typename OT, typename=std::enable_if_t<!dm::is_distance_matrix_v<OT> && !blaze::IsDenseMatrix_v<OT> && !blaze::IsSparseMatrix_v<OT>>>
void fill_helper(OT &) {
}

template<typename MT>
void fill_symmetric_upper_triangular(MT &mat) {
    fill_helper(mat);
}
using namespace blaze;



template<typename MT, bool SO>
void normalize(Matrix<MT, SO> &mat, bool rowwise=IsRowMajorMatrix_v<MT>) {
    if(rowwise) {
        for(auto r: rowiterator(~mat)) {
            auto n = l2Norm(r);
            if(n) r /= l2Norm(r);
        }
    } else {
        for(auto r: columniterator(~mat)) {
            auto n = l2Norm(r);
            if(n) r /= l2Norm(r);
        }
    }
}

#ifndef NDEBUG
#define assert_all_nonzero(...)
#else
#define assert_all_nonzero(x) do {::blz::_assert_all_nonzero_(x, __PRETTY_FUNCTION__, __FILE__, __LINE__);} while(0)
#endif

} // namespace blz
