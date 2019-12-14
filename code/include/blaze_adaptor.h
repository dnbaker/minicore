#pragma once
#include "blaze/Math.h"
#include "macros.h"

namespace blz {

// These blaze adaptors exist for the purpose of
// providing a pair of iterators.
template<typename this_type>
struct row_iterator_t {
    size_t rownum;
    this_type &ref_;

    auto index() const {return rownum;}
    row_iterator_t &operator++() {++rownum; return *this;}
    row_iterator_t &operator--() {--rownum; return *this;}
    row_iterator_t operator++(int) {
        row_iterator_t ret{rownum, ref_};
        ++rownum;
        return ret;
    }
    row_iterator_t operator--(int) {
        row_iterator_t ret{rownum, ref_};
        --rownum;
        return ret;
    }
    bool operator==(row_iterator_t o) const {return o.rownum == rownum;}
    bool operator!=(row_iterator_t o) const {return o.rownum != rownum;}
    bool operator<(row_iterator_t o) const {return o.rownum < rownum;}
    bool operator<=(row_iterator_t o) const {return o.rownum <= rownum;}
    bool operator>(row_iterator_t o) const {return o.rownum > rownum;}
    bool operator>=(row_iterator_t o) const {return o.rownum >= rownum;}
    std::ptrdiff_t operator-(row_iterator_t o) const {return rownum - o.rownum;}
    auto operator[](size_t index) const {
        return row(ref_, index + rownum);
    }
    auto operator*() const {
        return row(ref_, rownum);
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
    bool operator<(column_iterator_t o) const {return o.columnnum < columnnum;}
    bool operator<=(column_iterator_t o) const {return o.columnnum <= columnnum;}
    bool operator>(column_iterator_t o) const {return o.columnnum > columnnum;}
    bool operator>=(column_iterator_t o) const {return o.columnnum >= columnnum;}
    std::ptrdiff_t operator-(column_iterator_t o) const {return columnnum - o.columnnum;}
    auto operator[](size_t index) const {
        return column(ref_, index + columnnum);
    }
    auto operator*() const {
        return column(ref_, columnnum);
    }
};


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
    struct RowViewer {
        row_iterator start_, end_;
        RowViewer(this_type &ref): start_{0, ref}, end_{ref.rows(), ref} {}
        auto begin() const {return start_;}
        const auto &end()  const {return end_;}
    };
    struct ConstRowViewer {
        const_row_iterator start_, end_;
        ConstRowViewer(const this_type &ref): start_{0, ref}, end_{ref.rows(), ref} {}
        auto begin() const {return start_;}
        const auto &end()  const {return end_;}
    };
    struct ColumnViewer {
        auto index() const {return start_.columnnum;}
        column_iterator start_, end_;
        ColumnViewer(this_type &ref): start_{0, ref}, end_{ref.columns(), ref} {}
        auto begin() const {return start_;}
        const auto &end()  const {return end_;}
    };
    struct ConstColumnViewer {
        auto index() const {return start_.columnnum;}
        column_iterator start_, end_;
        ConstColumnViewer(const this_type &ref): start_{0, ref}, end_{ref.columns(), ref} {}
        auto begin() const {return start_;}
        const auto &end()  const {return end_;}
    };
    auto rowiterator()       {return RowViewer(*this);}
    auto rowiterator() const {return ConstRowViewer(*this);}
    auto columniterator()       {return ColumnViewer(*this);}
    auto columniterator() const {return ConstColumnViewer(*this);}
};

template< typename Type, bool AF, bool PF, bool SO >
class CustomMatrix: public blaze::CustomMatrix<Type, AF, PF, SO> {
    using super = blaze::CustomMatrix<Type, AF, PF, SO>;
    using this_type = CustomMatrix<Type, AF, PF, SO>;
    template<typename...Args>
    CustomMatrix(Args &&...args): super(std::forward<Args>(args)...) {}
    struct row_iterator: public row_iterator_t<this_type> {};
    struct const_row_iterator: public row_iterator_t<const this_type> {};
    struct column_iterator: public column_iterator_t<this_type> {};
    struct const_column_iterator: public column_iterator_t<const this_type> {};
    struct RowViewer {
        row_iterator start_, end_;
        RowViewer(this_type &ref): start_{0, ref}, end_{ref.rows(), ref} {}
        auto begin() const {return start_;}
        auto &end()  const {return end_;}
    };
    struct ConstRowViewer {
        const_row_iterator start_, end_;
        ConstRowViewer(const this_type &ref): start_{0, ref}, end_{ref.rows(), ref} {}
        auto begin() const {return start_;}
        auto &end()  const {return end_;}
    };
    struct ColumnViewer {
        auto index() const {return start_.columnnum;}
        column_iterator start_, end_;
        ColumnViewer(this_type &ref): start_{0, ref}, end_{ref.columns(), ref} {}
        auto begin() const {return start_;}
        auto &end()  const {return end_;}
    };
    struct ConstColumnViewer {
        auto index() const {return start_.columnnum;}
        column_iterator start_, end_;
        ConstColumnViewer(const this_type &ref): start_{0, ref}, end_{ref.columns(), ref} {}
        auto begin() const {return start_;}
        auto &end()  const {return end_;}
    };
    auto rowiterator()       {return RowViewer(*this);}
    auto rowiterator() const {return ConstRowViewer(*this);}
    auto columniterator()       {return ColumnViewer(*this);}
    auto columniterator() const {return ConstColumnViewer(*this);}
};

#define DECL_DIST(norm) \
template<typename FT, bool SO>\
INLINE double norm##Dist(const blaze::DynamicVector<FT, SO> &lhs, const blaze::DynamicVector<FT, SO> &rhs) {\
    return norm##Norm(rhs - lhs);\
}\
\
template<typename FT, bool AF, bool PF, bool SO, bool OAF, bool OPF>\
INLINE double norm##Dist(const blaze::CustomVector<FT, AF, PF, SO> &lhs,\
                      const blaze::CustomVector<FT, OAF, OPF, SO> &rhs)\
{\
    return norm##Norm(rhs - lhs);\
}\
\
template<typename FT, bool AF, bool PF, bool SO, bool OAF, bool OPF>\
INLINE double norm##Dist(const blaze::CustomVector<FT, AF, PF, SO> &lhs,\
                      const blaze::CustomVector<FT, OAF, OPF, !SO> &rhs)\
{\
    return norm##Norm(rhs - trans(lhs));\
}\
\
template<typename MatType1, typename MatType2, bool SO, bool isDense, bool isDense2, bool isSymmetric>\
INLINE double norm##Dist(const blaze::Row<MatType1, SO, isDense, isSymmetric> &lhs,\
                      const blaze::Row<MatType2, SO, isDense2, isSymmetric> &rhs)\
{\
    return norm##Norm(rhs - lhs);\
}\
\
template<typename MatType1, typename MatType2, bool SO, bool isDense, bool isDense2, bool isSymmetric>\
INLINE double norm##Dist(const blaze::Column<MatType1, SO, isDense, isSymmetric> &lhs,\
                      const blaze::Column<MatType2, SO, isDense2, isSymmetric> &rhs)\
{\
    return norm##Norm(rhs - lhs);\
}
DECL_DIST(l1)
DECL_DIST(l2)
DECL_DIST(sqr)
DECL_DIST(l3)
DECL_DIST(l4)
DECL_DIST(max)
DECL_DIST(inf)
#undef DECL_DIST

// TODO: replace manual, naive implementations with blaze::CustomVector.
template<typename FT, typename A, typename OA>
inline double l2Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    assert(lhs.size() == rhs.size());
    double s = 0.;
    for(size_t i = 0; i < lhs.size(); ++i) {
        FT tmp = lhs[i] - rhs[i];
        s += tmp * tmp;
    }
    return s;
}

template<typename FT, typename A, typename OA>
inline double l1Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    assert(lhs.size() == rhs.size());
    double ret = 0.;
    for(size_t i = 0; i < lhs.size(); ++i) {
        ret += std::abs(lhs[i] - rhs[i]);
    }
    return ret;
}
template<typename FT, typename A, typename OA>
inline double l3Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    assert(lhs.size() == rhs.size());
    double ret = 0.;
    for(size_t i = 0; i < lhs.size(); ++i) {
        ret += std::pow(std::abs(lhs[i] - rhs[i]), 3.);
    }
    return ret;
}
template<typename FT, typename A, typename OA>
inline double l4Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    assert(lhs.size() == rhs.size());
    double ret = 0.;
    for(size_t i = 0; i < lhs.size(); ++i) {
        ret += std::pow(lhs[i] - rhs[i], 4.);
    }
    return ret;
}
template<typename FT, typename A, typename OA>
inline double maxDist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    assert(lhs.size() == rhs.size());
    double ret = 0.;
    for(size_t i = 0; i < lhs.size(); ++i) {
        ret = std::max(std::abs(lhs[i] - rhs[i]));
    }
    return ret;
}
template<typename FT, typename A, typename OA>
inline double infDist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {return maxDist(lhs, rhs);}

struct L1Norm {
    template<typename C1, typename C2>
    constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        return l1Dist(lhs, rhs);
    }
};
struct L2Norm {
    template<typename C1, typename C2>
    constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        return l2Dist(lhs, rhs);
    }
};
struct L3Norm {
    template<typename C1, typename C2>
    constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        return l3Dist(lhs, rhs);
    }
};
struct L4Norm {
    template<typename C1, typename C2>
    constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        return l4Dist(lhs, rhs);
    }
};
struct maxNormFunctor {
    template<typename C1, typename C2>
    constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        return maxDist(lhs, rhs);
    }
};
struct infNormFunction: maxNormFunctor{};

using namespace blaze;
}
