#pragma once
#if defined(__has_include) && __has_include("sleef.h")
  extern "C" {
#if 0
#  if INLINE_SLEEF
#    include <limits.h>
#    include <float.h>
#    ifdef __GNUC__
#      define SLEEF_ALWAYS_INLINE __attribute((always_inline)) inline
#    else
#      define SLEEF_ALWAYS_INLINE
#    endif
#    define SLEEF_INLINE inline
#    define SLEEF_CONST __attribute__((const))
#    ifdef __AVX512F__
#      ifdef __AVX5124FMAPS__
#        include "sleefinline_avx512f.h"
#      else
#        include "sleefinline_avx512fnofma.h"
#      endif
#    elif __AVX__
#    include "sleefinline_avx.h"
#    elif __SSE4__
#    include "sleefinline_sse4.h"
#    elif __SSE2__
#    include "sleefinline_sse2.h"
#    else
#    include "sleefinline_purecfma_scalar.h"
#    endif
#  else
#    error("hello")
#    include "sleef.h"
#  endif
#else
#   include "sleef.h"
#endif
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
        assert(index + rownum < ref_.rows() || !std::fprintf(stderr, "Attempting to access %zu + %zu = %zu / %zu\n", index, rownum, index + rownum, ref_.rows()));
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
    row_iterator_t<MatType> start_, end_;
    RowViewer(MatType &mat): start_(0, mat), end_(mat.rows(), mat) {}
    auto begin() const {return start_;}
    auto end()   const {return end_;}
    template<size_t I>
    auto &get() const {
        if constexpr(I == 0) {
            return start_;
        } else {
            return end_;
        }
    }
};


template<typename MatType>
struct ColumnViewer {
    column_iterator_t<MatType> start_, end_;
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

#define DOFUNC(fn) auto fn() const {return (**this).fn();}
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
    decltype(auto) operator[](size_t i) const {
        if constexpr(SO == blaze::rowMajor) {
            return row(*this, i, blaze::unchecked);
        } else {
            return column(*this, i, blaze::unchecked);
        }
    }
    decltype(auto) operator[](size_t i) {
        if constexpr(SO == blaze::rowMajor) {
            return row(*this, i, blaze::unchecked);
        } else {
            return column(*this, i, blaze::unchecked);
        }
    }
    template<typename...Args> this_type &operator=(Args &&...args) {
        ((super &)*this).operator=(std::forward<Args>(args)...);
        return *this;
    }
    size_t size() const {return SO == blaze::rowMajor ? this->rows(): this->columns();}
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
    decltype(auto) operator[](size_t i) const {
        if constexpr(SO == blaze::rowMajor) {
            return row(*this, i, blaze::unchecked);
        } else {
            return column(*this, i, blaze::unchecked);
        }
    }
    decltype(auto) operator[](size_t i) {
        if constexpr(SO == blaze::rowMajor) {
            return row(*this, i, blaze::unchecked);
        } else {
            return column(*this, i, blaze::unchecked);
        }
    }
    size_t size() const {return SO == blaze::rowMajor ? this->rows(): this->columns();}
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

template<typename FT, typename Alloc>
INLINE auto sum(const std::vector<FT, Alloc> &vec) {
    return blaze::sum(blaze::CustomVector<FT, blaze::unaligned, blaze::unpadded>(const_cast<FT *>(vec.data()), vec.size()));
}
template<typename FT, typename Alloc>
INLINE auto max(const std::vector<FT, Alloc> &vec) {
    return blaze::max(blaze::CustomVector<FT, blaze::unaligned, blaze::unpadded>(const_cast<FT *>(vec.data()), vec.size()));
}
template<typename FT, typename Alloc>
INLINE auto min(const std::vector<FT, Alloc> &vec) {
    return blaze::min(blaze::CustomVector<FT, blaze::unaligned, blaze::unpadded>(const_cast<FT *>(vec.data()), vec.size()));
}
template<typename FT, typename Alloc>
INLINE auto mean(const std::vector<FT, Alloc> &vec) {
    return blaze::mean(blaze::CustomVector<FT, blaze::unaligned, blaze::unpadded>(const_cast<FT *>(vec.data()), vec.size()));
}

template<typename OT>
INLINE decltype(auto) sum(const OT &x) {return blaze::sum(x);}

template<bool wiseness, typename OT>
INLINE decltype(auto) sum(const OT &x) {return blaze::sum<wiseness>(x);}

template<typename OT>
INLINE decltype(auto) max(const OT &x) {return blaze::max(x);}
template<bool wiseness, typename OT>
INLINE decltype(auto) max(const OT &x) {return blaze::max<wiseness>(x);}
template<typename OT>
INLINE decltype(auto) mean(const OT &x) {return blaze::mean(x);}
template<bool wiseness, typename OT>
INLINE decltype(auto) mean(const OT &x) {return blaze::mean<wiseness>(x);}
template<typename OT>
INLINE decltype(auto) min(const OT &x) {return blaze::min(x);}
template<bool wiseness, typename OT>
INLINE decltype(auto) min(const OT &x) {return blaze::min<wiseness>(x);}
template<typename...Args>
INLINE decltype(auto) min(Args &&...args) {return blaze::min(std::forward<Args>(args)...);}
template<typename...Args>
INLINE decltype(auto) max(Args &&...args) {return blaze::max(std::forward<Args>(args)...);}
template<typename...Args>
INLINE decltype(auto) mean(Args &&...args) {return blaze::mean(std::forward<Args>(args)...);}


template<typename VT, bool SO, typename VT2, bool SO2>
size_t number_shared_zeros(const blaze::SparseVector<VT, SO> &_lhs, const blaze::SparseVector<VT2, SO2> &_rhs) {
     auto &lhs = *_lhs;
     auto &rhs = *_rhs;
     assert(lhs.size() == rhs.size());
     //const size_t sz = lhs.size();
     auto lhit = lhs.begin();
     auto rhit = rhs.begin();
     auto lhe = lhs.end();
     auto rhe = rhs.end();
     if(lhit == lhe) return nonZeros(rhs);
     if(rhit == rhe) return nonZeros(lhs);
     auto getnextindex = [&]() {
         size_t r1 = lhit == lhe ? size_t(-1): lhit->index();
         size_t r2 = rhit == rhe ? size_t(-1): rhit->index();
         if(r1 == r2) {
             ++lhit;
             ++rhit;
         } else if(r1 < r2) ++lhit;
         else ++rhit;
         return std::min(r1, r2);
     };
     size_t current_index = getnextindex();
     size_t ret = current_index;
     for(size_t nv; (nv = getnextindex()) != size_t(-1);) {
         if(nv == current_index) continue;
         assert(nv > current_index);
         ret += nv - current_index - 1;
         current_index = nv;
     }
     return ret;
 }

template<typename MT, bool SO>
void fill_helper(blaze::Matrix<MT, SO> &mat) {
    diagonal(*mat) = 0.;
    const size_t nr = (*mat).rows();
    for(size_t i = 0; i < nr - 1; ++i) {
        submatrix(*mat, i + 1, i, nr - i - 1, 1) = trans(submatrix(*mat, i, i + 1, 1, nr - i - 1));
    }
}

template<typename FT, size_t DV, dm::MemoryStrategy ms>
void fill_helper(dm::DistanceMatrix<FT, DV, ms> &) {
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
        for(auto r: rowiterator(*mat)) {
            auto n = l2Norm(r);
            if(n) r /= l2Norm(r);
        }
    } else {
        for(auto r: columniterator(*mat)) {
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

#define DECL_DIST(norm) \
template<typename FT, bool SO>\
INLINE auto norm##Dist(const blaze::DynamicVector<FT, SO> &lhs, const blaze::DynamicVector<FT, SO> &rhs) {\
    return norm##Norm(rhs - lhs);\
}\
template<typename VT, typename VT2, bool SO>\
INLINE auto norm##Dist(const blaze::DenseVector<VT, SO> &lhs, const blaze::DenseVector<VT2, SO> &rhs) {\
    return norm##Norm(*rhs - *lhs);\
}\
\
template<typename VT, typename VT2, bool SO>\
INLINE auto norm##Dist(const blaze::DenseVector<VT, SO> &lhs, const blaze::DenseVector<VT2, !SO> &rhs) {\
    return norm##Norm(*rhs - trans(*lhs));\
}\
\
template<typename VT, typename VT2, bool SO>\
INLINE auto norm##Dist(const blaze::SparseVector<VT, SO> &lhs, const blaze::SparseVector<VT2, SO> &rhs) {\
    return norm##Norm(*rhs - *lhs);\
}\
\
template<typename VT, typename VT2, bool SO>\
INLINE auto norm##Dist(const blaze::SparseVector<VT, SO> &lhs, const blaze::SparseVector<VT2, !SO> &rhs) {\
    return norm##Norm(*rhs - trans(*lhs));\
}\

DECL_DIST(l1)
DECL_DIST(l2)
DECL_DIST(sqr)
DECL_DIST(l3)
DECL_DIST(l4)
DECL_DIST(max)
DECL_DIST(inf)
#undef DECL_DIST
template<typename FT, typename A, typename OA>
inline auto l2Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return l2Dist(CustomVector<const FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<const FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}

template<typename FT, typename FT2, bool SO>
inline auto sqrL2Dist(const blz::Vector<FT, SO> &v1, const blz::Vector<FT2, SO> &v2) {
    return sqrDist(*v1, *v2);
}
template<typename FT, blaze::AlignmentFlag AF, blaze::PaddingFlag PF, bool SO, blaze::AlignmentFlag OAF, blaze::PaddingFlag OPF, bool OSO>
inline auto sqrL2Dist(const blz::CustomVector<FT, AF, PF, SO> &v1, const blz::CustomVector<FT, OAF, OPF, OSO> &v2) {
    return sqrDist(v1, v2);
}

template<typename FT, typename A, typename OA>
inline auto sqrL2Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return sqrL2Dist(CustomVector<const FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                     CustomVector<const FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}

template<typename FT, typename A, typename OA>
INLINE auto sqrDist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return sqrL2Dist(lhs, rhs);
}

template<typename FT, typename A, typename OA>
inline auto l1Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return l1Dist(CustomVector<const FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<const FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}
template<typename FT, typename A, typename OA>
inline auto l3Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return l3Dist(CustomVector<const FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<const FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}

template<typename FT, typename A, typename OA>
inline auto l4Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return l4Dist(CustomVector<FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}
template<typename FT, typename A, typename OA>
inline auto maxDist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return maxDist(CustomVector<FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}
template<typename FT, typename A, typename OA>
inline auto infDist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {return maxDist(lhs, rhs);}

template<typename Base>
struct sqrBaseNorm: public Base {
    template<typename C1, typename C2>
    INLINE constexpr auto operator()(const C1 &lhs, const C2 &rhs) const {
        return std::pow(Base::operator()(lhs, rhs), 2);
    }
};
struct L1Norm {
    template<typename C1, typename C2>
    INLINE constexpr auto operator()(const C1 &lhs, const C2 &rhs) const {
        return l1Dist(lhs, rhs);
    }
};
struct L2Norm {
    template<typename C1, typename C2>
    INLINE constexpr auto operator()(const C1 &lhs, const C2 &rhs) const {
        return l2Dist(lhs, rhs);
    }
};
struct sqrL2Norm {
    template<typename C1, typename C2>
    INLINE constexpr auto operator()(const C1 &lhs, const C2 &rhs) const {
        return sqrDist(lhs, rhs);
    }
};
struct L3Norm {
    template<typename C1, typename C2>
    INLINE constexpr auto operator()(const C1 &lhs, const C2 &rhs) const {
        return l3Dist(lhs, rhs);
    }
};
struct L4Norm {
    template<typename C1, typename C2>
    INLINE constexpr auto operator()(const C1 &lhs, const C2 &rhs) const {
        return l4Dist(lhs, rhs);
    }
};
struct maxNormFunctor {
    template<typename C1, typename C2>
    INLINE constexpr auto operator()(const C1 &lhs, const C2 &rhs) const {
        return maxDist(lhs, rhs);
    }
};
struct infNormFunction: maxNormFunctor{};
struct sqrL1Norm: sqrBaseNorm<L1Norm> {};
struct sqrL3Norm: sqrBaseNorm<L3Norm> {};
struct sqrL4Norm: sqrBaseNorm<L4Norm> {};
struct sqrMaxNorm: sqrBaseNorm<maxNormFunctor> {};


// For D^2 sampling.
template<typename BaseDist>
struct SqrNormFunctor: public BaseDist {
    template<typename...Args> SqrNormFunctor(Args &&...args): BaseDist(std::forward<Args>(args)...) {}
    template<typename C1, typename C2>
    INLINE constexpr auto operator()(const C1 &lhs, const C2 &rhs) const {
        auto basedist = BaseDist::operator()(lhs, rhs);
        return basedist * basedist;
    }
};
template<>
struct SqrNormFunctor<L2Norm>: public sqrL2Norm {};

namespace functional {

template<typename F, typename IT=std::uint32_t, size_t N=16>
auto indices_if(const F &func, size_t n) {
    blaze::SmallArray<IT, N> ret;
    for(IT i = 0; i < n; ++i) if(func(i)) ret.pushBack(i);
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


} // namespace blz::functional
using functional::indices_if;


template<typename FT, typename IT>
auto make_cv(FT *data, IT size) {
    using T = std::remove_const_t<FT>;
    using BaseRetT = CustomVector<T, unaligned, unpadded>;
    using RetT = std::conditional_t<std::is_const_v<FT>, const BaseRetT, BaseRetT>;
    return RetT(const_cast<T *>(data), size);
}



// Solve geometric median for a set of points.
template<typename MT, bool SO, typename VT, typename WeightType>
auto &geomedian(const Matrix<MT, SO> &mat, Vector<VT, !SO> &dv, WeightType *const weights, double eps=0)
{
    if((*mat).rows() == 1) return *dv = row((*mat), 0);
    const auto &_mat = *mat;
    using FT = typename std::decay_t<decltype(*mat)>::ElementType;
    FT prevcost = std::numeric_limits<FT>::max();
    size_t iternum = 0;
    assert((*dv).size() == (*mat).columns());
    DV<FT, SO> costs(_mat.rows());
    std::unique_ptr<CustomVector<WeightType, unaligned, unpadded, SO>> cv;
    if(weights)
        cv.reset(new CustomVector<WeightType, unaligned, unpadded, SO>(const_cast<WeightType *>(weights), _mat.rows()));
    for(;;) {
#ifndef NDEBUG
        std::fprintf(stderr, "Iteration %zu for matrix %zu/%zu and vector %zu with weights at %p\n",
                     iternum + 1, (*mat).rows(), (*mat).columns(), (*dv).size(), (void *)weights);
#endif
        if(weights) {
            auto &cvr = *cv;
            OMP_PFOR
            for(size_t i = 0; i < _mat.rows(); ++i)
                costs[i] = cvr[i] * blz::l2Norm(row(_mat, i, blaze::unchecked) - *dv);
        } else {
#if 1
            OMP_PFOR
            for(size_t i = 0; i < _mat.rows(); ++i) {
#if 0
                const auto r = row(_mat, i, blaze::unchecked) - *dv;
                std::cerr << "row #" << i << " is " << r << '\n';
                std::cerr << "cost before is: " << costs[i] << '\n';
                std::cerr << "Row: " << row(_mat, i) << '\n';
                std::cerr << "center: " << *dv << '\n';
                costs[i] = blz::sqrt(blz::sum(r * r));
                std::cerr << "cost after is " << costs[i] << '\n';
                if(std::isnan(costs[i])) {
                    std::cerr << "cost after is NAN: " << costs[i] << '\n';
                    std::cerr << r << '\n';
                    std::cerr << "r squared should be " << (r * r) << '\n';
                }
#else
                using res_t = std::decay_t<decltype(blz::l2Norm(row(_mat, i, blz::unchecked) - *dv))>;
                costs[i] = std::max(blz::l2Norm(row(_mat, i, blz::unchecked) - *dv),
                                    static_cast<res_t>(1e-80));
#endif
            }
#else
            costs = sqrt(sum<rowwise>(blz::pow(_mat - blaze::expand(*dv, (*mat).rows()), 2))); // pow2 seems broken
#endif
        }
        FT current_cost = sum(costs);
        FT dist;
        if((dist = std::abs(prevcost - current_cost)) <= eps) break;
        if(unlikely(std::isnan(dist))) {
            std::fprintf(stderr, "[%s:%s:%d] dist is nan\n", __PRETTY_FUNCTION__, __FILE__, __LINE__);
            throw std::runtime_error("Optimization failed: nan");
        }
        ++iternum;
        costs = 1. / costs;
        costs *= 1. / blaze::sum(costs);
        *dv = trans(costs) * *mat;
        prevcost = current_cost;
    }
    return *dv;
}
// Solve geometric median for a set of points.
template<typename MT, bool SO, typename VT, typename WeightType, typename=std::enable_if_t<!std::is_arithmetic_v<WeightType>>>
auto &geomedian(const Matrix<MT, SO> &mat, Vector<VT, !SO> &dv, const WeightType &weights, double eps=0.)
{
    if((*mat).rows() == 1) return *dv = row((*mat), 0);
    const auto &_mat = *mat;
    using FT = typename std::decay_t<decltype(*mat)>::ElementType;
    FT prevcost = std::numeric_limits<FT>::max();
    size_t iternum = 0;
    assert((*dv).size() == (*mat).columns());
    DV<FT, SO> costs(_mat.rows());
    std::unique_ptr<CustomVector<WeightType, unaligned, unpadded, SO>> cv;
    for(;;) {
        OMP_PFOR
        for(size_t i = 0; i < _mat.rows(); ++i) {
            using res_t = std::decay_t<decltype(weights[0] * blz::l2Norm(row(_mat, 0) - *dv))>;
            costs[i] = std::max(weights[i] * blz::l2Norm(row(_mat, i, blaze::unchecked) - *dv), res_t(1e-80));
        }
        FT current_cost = sum(costs);
        FT dist;
        if((dist = std::abs(prevcost - current_cost)) <= eps) break;
        if(unlikely(std::isnan(dist))) {
            std::fprintf(stderr, "[%s:%s:%d] dist is nan\n", __PRETTY_FUNCTION__, __FILE__, __LINE__);
            throw std::runtime_error("Optimization failed: nan");
            break;
        }
        ++iternum;
        costs = 1. / costs;
        costs *= 1. / blaze::sum(costs);
        *dv = trans(costs) * *mat;
        prevcost = current_cost;
    }
    return *dv;
}
template<typename MT, bool SO, typename VT>
auto &geomedian(const Matrix<MT, SO> &mat, Vector<VT, !SO> &dv, double eps=0) {
    return geomedian<MT, SO, VT, blz::ElementType_t<MT>>(mat, dv, static_cast<blaze::ElementType_t<MT> *>(nullptr), eps);
}

} // namespace blz


namespace std {

#define OLOAD(MAT, N)\
   template<typename this_type> struct tuple_element<N, MAT<this_type>> {\
        using type = decltype(std::declval<MAT<this_type>>().get<N>());\
   };
   template<size_t N, typename MatrixType>
   struct tuple_element<N, blz::ColumnViewer<MatrixType>> {
       using type = std::decay_t<decltype(std::declval<blz::ColumnViewer<MatrixType>>().template get<N>())>;
   };
   template<size_t N, typename MatrixType>
   struct tuple_element<N, blz::RowViewer<MatrixType>> {
       using type = std::decay_t<decltype(std::declval<blz::RowViewer<MatrixType>>().template get<N>())>;
   };
#if 0
   template<typename this_type> struct tuple_element<0,blz::ConstColumnViewer<this_type>> { using type = typename blz::column_iterator_t<this_type> &; };
   template<typename this_type> struct tuple_element<0,blz::ColumnViewer<this_type>> { using type = typename blz::column_iterator_t<this_type> &; };
   template<typename this_type> struct tuple_element<0,blz::ConstRowViewer<this_type>> { using type = typename blz::row_iterator_t<this_type> &; };
   template<typename this_type> struct tuple_element<0,blz::RowViewer<this_type>> { using type = typename blz::row_iterator_t<this_type> &; };
   template<typename this_type> struct tuple_element<1,blz::ConstColumnViewer<this_type>> { using type = typename blz::column_iterator_t<this_type> &; };
   template<typename this_type> struct tuple_element<1,blz::ColumnViewer<this_type>> { using type = typename blz::column_iterator_t<this_type> &; };
   template<typename this_type> struct tuple_element<1,blz::ConstRowViewer<this_type>> { using type = typename blz::row_iterator_t<this_type> &; };
   template<typename this_type> struct tuple_element<1,blz::RowViewer<this_type>> { using type = typename blz::row_iterator_t<this_type> &; };
#endif

   template<typename this_type> struct tuple_size<blz::RowViewer<this_type>> : public std::integral_constant<size_t,2> {};
   template<typename this_type> struct tuple_size<blz::ColumnViewer<this_type>>: public std::integral_constant<size_t,2> {};
   template<int I, typename Mat>
   decltype(auto) get(const blz::RowViewer<Mat> &x) {
        return x.get();
   }
   template<int I, typename Mat>
   decltype(auto) get(const blz::ConstRowViewer<Mat> &x) {return x.get();}
   template<int I, typename Mat>
   decltype(auto) get(const blz::ColumnViewer<Mat> &x) {return x.get();}
   template<int I, typename Mat>
   decltype(auto) get(const blz::ConstColumnViewer<Mat> &x) {return x.get();}
} // namespace std

