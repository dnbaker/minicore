#pragma once
#include "blaze/Math.h"
#include <vector>
#include "./shared.h"

namespace blz {
using blaze::unchecked;

// These blaze adaptors exist for the purpose of
// providing a pair of iterators.
template<typename this_type>
struct row_iterator_t {
    size_t rownum;
    this_type &ref_;

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
    RowViewer(MatType &mat): start_{{0, mat}}, end_{{mat.rows(), mat}} {}
    auto begin() const {return start_;}
    auto end()   const {return end_;}
};


template<typename MatType>
struct ColumnViewer {
    const column_iterator_t<MatType> start_, end_;
    ColumnViewer(MatType &mat): start_{{0, mat}}, end_{{mat.columns(), mat}} {}
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
    DOFUNC(size)\
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
#if 0
    struct RowViewer {
        row_iterator start_, end_;
        RowViewer(this_type &ref): start_{{0, ref}}, end_{{ref.rows(), ref}} {}
        auto begin() const {return start_;}
        const auto &end()  const {return end_;}
    };
    struct ConstRowViewer {
        const_row_iterator start_, end_;
        ConstRowViewer(const this_type &ref): start_{{0, ref}}, end_{{ref.rows(), ref}} {}
        auto begin() const {return start_;}
        const auto &end()  const {return end_;}
    };
    struct ColumnViewer {
        auto index() const {return start_.columnnum;}
        column_iterator start_, end_;
        ColumnViewer(this_type &ref): start_{{0, ref}}, end_{{ref.columns(), ref}} {}
        auto begin() const {return start_;}
        const auto &end()  const {return end_;}
    };
    struct ConstColumnViewer {
        auto index() const {return start_.columnnum;}
        const_column_iterator start_, end_;
        ConstColumnViewer(const this_type &ref): start_{{0, ref}}, end_{{ref.columns(), ref}} {}
        auto begin() const {return start_;}
        const auto &end()  const {return end_;}
    };
#endif
    auto rowiterator()       {return RowViewer<this_type>(*this);}
    auto rowiterator() const {return ConstRowViewer<this_type>(*this);}
    auto columniterator()       {return ColumnViewer<this_type>(*this);}
    auto columniterator() const {return ConstColumnViewer<this_type>(*this);}
    ADD_FUNCS
};


template<typename FT, bool SO>
auto rowiterator(blaze::DynamicMatrix<FT, SO> &o) {
    return reinterpret_cast<blz::DynamicMatrix<FT, SO> &>(o).rowiterator();
}

template<typename FT, bool SO>
auto rowiterator(const blaze::DynamicMatrix<FT, SO> &o) {
    return reinterpret_cast<const blz::DynamicMatrix<FT, SO> &>(o).rowiterator();
}

#if 0
template<typename MT, bool AF, bool SO, bool DF>
auto rowiterator(const blaze::Submatrix<MT, AF, SO, DF> &o) {
    return 
}
#endif



template<typename FT, bool SO>
auto columniterator(blaze::DynamicMatrix<FT, SO> &o) {
    return reinterpret_cast<blz::DynamicMatrix<FT, SO> &>(o).columniterator();
}
template<typename FT, bool SO>
auto columniterator(const blaze::DynamicMatrix<FT, SO> &o) {
    return reinterpret_cast<const blz::DynamicMatrix<FT, SO> &>(o).columniterator();
}

template< typename Type, bool AF, bool PF, bool SO >
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
#if 0
    struct RowViewer {
        row_iterator start_, end_;
        RowViewer(this_type &ref): start_{{0, ref}}, end_{ref.rows(), ref} {}
        auto begin() const {return start_;}
        auto &end()  const {return end_;}
    };
    struct ConstRowViewer {
        const_row_iterator start_, end_;
        ConstRowViewer(const this_type &ref): start_{{0, ref}}, end_{ref.rows(), ref} {}
        auto begin() const {return start_;}
        auto &end()  const {return end_;}
    };
    struct ColumnViewer {
        auto index() const {return start_.columnnum;}
        column_iterator start_, end_;
        ColumnViewer(this_type &ref): start_{{0, ref}}, end_{ref.columns(), ref} {}
        auto begin() const {return start_;}
        auto &end()  const {return end_;}
    };
    struct ConstColumnViewer {
        auto index() const {return start_.columnnum;}
        column_iterator start_, end_;
        ConstColumnViewer(const this_type &ref): start_{{0, ref}}, end_{ref.columns(), ref} {}
        auto begin() const {return start_;}
        auto &end()  const {return end_;}
    };
#endif
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
template<typename FT, bool AF=blaze::unaligned, bool PF=blaze::unpadded, bool SO=blaze::rowMajor>
using CM = CustomMatrix<FT, AF, PF, SO>;
template<typename FT, bool AF=blaze::unaligned, bool PF=blaze::unpadded, bool TF=blaze::columnVector>
using CV = blaze::CustomVector<FT, AF, PF, TF>;


#if 0
template<typename FT, bool AF, bool PF, bool SO>
auto rowiterator(blaze::CustomMatrix<FT, AF, PF, SO> &o) {
    return reinterpret_cast<blz::CustomMatrix<FT, AF, PF, SO> &>(o).rowiterator();
}
template<typename FT, bool AF, bool PF, bool SO>
auto rowiterator(const blaze::CustomMatrix<FT, AF, PF, SO> &o) {
    return reinterpret_cast<const blz::CustomMatrix<FT, AF, PF, SO> &>(o).rowiterator();
}
template<typename FT, bool AF, bool PF, bool SO>
auto columniterator(blaze::CustomMatrix<FT, AF, PF, SO> &o) {
    return reinterpret_cast<blz::CustomMatrix<FT, AF, PF, SO> &>(o).columniterator();
}
template<typename FT, bool AF, bool PF, bool SO>
auto columniterator(const blaze::CustomMatrix<FT, AF, PF, SO> &o) {
    return reinterpret_cast<const blz::CustomMatrix<FT, AF, PF, SO> &>(o).columniterator();
}
#endif

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
using namespace blaze;
} // namespace blz
