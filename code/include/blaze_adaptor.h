#pragma once
#include "blaze/Math.h"
namespace blz {

// These blaze adaptors exist for the purpose of
// providing a pair of iterators.
template<typename this_type>
struct row_iterator_t {
    auto index() const {return rownum;}
    size_t rownum;
    this_type &ref_;
    row_iterator_t &operator++() {++rownum; return *this;}
    bool operator==(row_iterator_t o) const {return o.rownum == rownum;}
    bool operator!=(row_iterator_t o) const {return o.rownum != rownum;}
    bool operator<(row_iterator_t o) const {return o.rownum < rownum;}
    bool operator<=(row_iterator_t o) const {return o.rownum <= rownum;}
    bool operator>(row_iterator_t o) const {return o.rownum > rownum;}
    bool operator>=(row_iterator_t o) const {return o.rownum >= rownum;}
    auto operator*() const {
        return row(ref_, rownum);
    }
};
template<typename this_type>
struct column_iterator_t {
    auto index() const {return columnnum;}
    size_t columnnum;
    this_type &ref_;
    column_iterator_t &operator++() {++columnnum; return *this;}
    bool operator==(column_iterator_t o) const {return o.columnnum == columnnum;}
    bool operator!=(column_iterator_t o) const {return o.columnnum != columnnum;}
    bool operator<(column_iterator_t o) const {return o.columnnum < columnnum;}
    bool operator<=(column_iterator_t o) const {return o.columnnum <= columnnum;}
    bool operator>(column_iterator_t o) const {return o.columnnum > columnnum;}
    bool operator>=(column_iterator_t o) const {return o.columnnum >= columnnum;}
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
template<typename FT, bool SO>
INLINE double sqrNorm(const blaze::DynamicMatrix<FT, SO> &lhs, const blaze::DynamicMatrix<FT, SO> &rhs) {
    return sqrNorm(rhs - lhs);
}

template<typename FT, bool AF, bool PF, bool SO, bool OAF, bool OPF>
INLINE double sqrNorm(const blaze::CustomMatrix<FT, AF, PF, SO> &lhs,
                      const blaze::CustomMatrix<FT, OAF, OPF, SO> &rhs)
{
    return sqrNorm(rhs - lhs);
}

template<typename FT, bool AF, bool PF, bool SO, bool OAF, bool OPF>
INLINE double sqrNorm(const blaze::CustomMatrix<FT, AF, PF, SO> &lhs,
                      const blaze::CustomMatrix<FT, OAF, OPF, !SO> &rhs)
{
    return sqrNorm(rhs - trans(lhs));
}
using namespace blaze;
}
