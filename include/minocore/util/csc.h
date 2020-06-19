#ifndef CSC_H__
#define CSC_H__
#include "./shared.h"
#include "./timer.h"
#include "./blaze_adaptor.h"
#include "mio/single_include/mio/mio.hpp"
#include <fstream>

namespace minocore {

namespace util {

template<typename IndPtrType=uint64_t, typename IndicesType=uint64_t, typename DataType=uint32_t>
struct CSCMatrixView {
    using ElementType = DataType;
    const IndPtrType *const indptr_;
    const IndicesType *const indices_;
    const DataType *const data_;
    const uint64_t nnz_;
    const uint32_t nf_, n_;
    static_assert(std::is_integral_v<IndPtrType>, "IndPtr must be integral");
    static_assert(std::is_integral_v<IndicesType>, "Indices must be integral");
    static_assert(std::is_arithmetic_v<IndicesType>, "Data must be arithmetic");
    CSCMatrixView(const IndPtrType *indptr, const IndicesType *indices, const DataType *data,
                  uint64_t nnz, uint32_t nfeat, uint32_t nitems):
        indptr_(indptr),
        indices_(indices),
        data_(data),
        nnz_(nnz),
        nf_(nfeat), n_(nitems)
    {
    }
    struct CView: public std::pair<DataType *, size_t> {
        size_t index() const {return this->second;}
        DataType &value() {return *this->first;}
        const DataType &value() const {return *this->first;}
    };
    struct ConstCView: public std::pair<const DataType*, size_t> {
        size_t index() const {return this->second;}
        const DataType &value() const {return *this->first;}
    };
    struct Column {
        const CSCMatrixView &mat_;
        size_t start_;
        size_t stop_;

        Column(const CSCMatrixView &mat, size_t start, size_t stop): mat_(mat), start_(start), stop_(stop) {}
        size_t nnz() const {return stop_ - start_;}
        size_t size() const {return mat_.columns();}
        template<bool is_const>
        struct ColumnIteratorBase {
            using ViewType = std::conditional_t<is_const, ConstCView, CView>;
            using ColType = std::conditional_t<is_const, const Column, Column>;
            using ViewedType = std::conditional_t<is_const, const DataType, DataType>;
            using difference_type = std::ptrdiff_t;
            using value_type = ViewedType;
            using reference = ViewedType &;
            using pointer = ViewedType *;
            using iterator_category = std::random_access_iterator_tag;
            ColType &col_;
            size_t index_;
            private:
            mutable ViewType data_;
            public:

            template<bool oconst>
            bool operator==(const ColumnIteratorBase<oconst> &o) const {
                return index_ == o.index_;
            }
            template<bool oconst>
            bool operator!=(const ColumnIteratorBase<oconst> &o) const {
                return index_ != o.index_;
            }
            template<bool oconst>
            bool operator<(const ColumnIteratorBase<oconst> &o) const {
                return index_ < o.index_;
            }
            template<bool oconst>
            bool operator>(const ColumnIteratorBase<oconst> &o) const {
                return index_ > o.index_;
            }
            template<bool oconst>
            bool operator<=(const ColumnIteratorBase<oconst> &o) const {
                return index_ <= o.index_;
            }
            template<bool oconst>
            bool operator>=(const ColumnIteratorBase<oconst> &o) const {
                return index_ >= o.index_;
            }
            template<bool oconst>
            difference_type operator-(const ColumnIteratorBase<oconst> &o) const {
                return this->index_ - o.index_;
            }
            ColumnIteratorBase<is_const> &operator++() {
                ++index_;
                return *this;
            }
            ColumnIteratorBase<is_const> operator++(int) {
                ColumnIteratorBase ret(col_, index_);
                ++index_;
                return ret;
            }
            const CView &operator*() const {
                set();
                return data_;
            }
            CView &operator*() {
                set();
                return data_;
            }
            void set() const {
                data_.first = const_cast<DataType *>(&col_.mat_.data_[index_]);
                data_.second = col_.mat_.indices_[index_];
            }
            ViewType *operator->() {
                set();
                return &data_;
            }
            const ViewType *operator->() const {
                set();
                return &data_;
            }
            ColumnIteratorBase(ColType &col, size_t ind): col_(col), index_(ind) {
            }
        };
        using ColumnIterator = ColumnIteratorBase<false>;
        using ConstColumnIterator = ColumnIteratorBase<true>;
        ColumnIterator begin() {return ColumnIterator(*this, start_);}
        ColumnIterator end()   {return ColumnIterator(*this, stop_);}
        ConstColumnIterator begin() const {return ConstColumnIterator(*this, start_);}
        ConstColumnIterator end()   const {return ConstColumnIterator(*this, stop_);}
    };
    auto column(size_t i) const {
        return Column(*this, indptr_[i], indptr_[i + 1]);
    }
    size_t nnz() const {return nnz_;}
    size_t rows() const {return n_;}
    size_t columns() const {return nf_;}
};

template<typename T>
struct is_csc_view: public std::false_type {};
template<typename IndPtrType, typename IndicesType, typename DataType>
struct is_csc_view<CSCMatrixView<IndPtrType, IndicesType, DataType>>: public std::true_type {};

template<typename T>
static constexpr bool is_csc_view_v = is_csc_view<T>::value;


template<typename DataType, typename IndPtrType, typename IndicesType>
size_t nonZeros(const typename CSCMatrixView<IndPtrType, IndicesType, DataType>::Column &col) {
    return col.nnz();
}
template<typename DataType, typename IndPtrType, typename IndicesType>
size_t nonZeros(const CSCMatrixView<IndPtrType, IndicesType, DataType> &mat) {
    return mat.nnz();
}

template<typename FT=float, typename IndPtrType, typename IndicesType, typename DataType>
blz::SM<FT, blaze::rowMajor> csc2sparse(const CSCMatrixView<IndPtrType, IndicesType, DataType> &mat, bool skip_empty=false) {
    blz::SM<FT, blaze::rowMajor> ret(mat.n_, mat.nf_);
    ret.reserve(mat.nnz_);
    size_t used_rows = 0, i;
    for(i = 0; i < mat.n_; ++i) {
        auto col = mat.column(i);
        if(mat.n_ > 100000 && i % 10000 == 0) std::fprintf(stderr, "%zu/%u\r", i, mat.n_);
        if(skip_empty && 0u == col.nnz()) continue;
        for(auto s = col.start_; s < col.stop_; ++s) {
            ret.append(used_rows, mat.indices_[s], mat.data_[s]);
        }
        ret.finalize(used_rows++);
    }
    if(used_rows != i) std::fprintf(stderr, "Only used %zu/%zu rows, skipping empty rows\n", used_rows, i);
    std::fprintf(stderr, "Parsed matrix of %zu/%zu\n", ret.rows(), ret.columns());
    return ret;
}

template<typename FT=float, typename IndPtrType=uint64_t, typename IndicesType=uint64_t, typename DataType=uint32_t>
blz::SM<FT, blaze::rowMajor> csc2sparse(std::string prefix, bool skip_empty=false) {
    util::Timer t("csc2sparse load time");
    std::string indptrn  = prefix + "indptr.file";
    std::string indicesn = prefix + "indices.file";
    std::string datan    = prefix + "data.file";
    std::string shape    = prefix + "shape.file";
    std::FILE *ifp = std::fopen(shape.data(), "rb");
    uint32_t dims[2];
    if(std::fread(dims, sizeof(uint32_t), 2, ifp) != 2) throw std::runtime_error("Failed to read dims from file");
    uint32_t nfeat = dims[0], nsamples = dims[1];
    std::fprintf(stderr, "nfeat: %u. nsample: %u\n", nfeat, nsamples);
    std::fclose(ifp);
    using mmapper = mio::mmap_source;
    mmapper indptr(indptrn), indices(indicesn), data(datan);
    CSCMatrixView<IndPtrType, IndicesType, DataType>
        matview((const IndPtrType *)indptr.data(), (const IndicesType *)indices.data(),
                (const DataType *)data.data(), indices.size() / sizeof(IndicesType),
                 nfeat, nsamples);
    std::fprintf(stderr, "indptr size: %zu\n", indptr.size() / sizeof(IndPtrType));
    std::fprintf(stderr, "indices size: %zu\n", indices.size() / sizeof(IndicesType));
    std::fprintf(stderr, "data size: %zu\n", data.size() / sizeof(DataType));
#ifndef MADV_REMOVE
#  define MADV_FLAGS (MADV_DONTNEED | MADV_FREE)
#else
#  define MADV_FLAGS (MADV_DONTNEED | MADV_REMOVE)
#endif
    ::madvise((void *)indptr.data(), indptr.size(), MADV_FLAGS);
    ::madvise((void *)indices.data(), indices.size(), MADV_FLAGS);
    ::madvise((void *)data.data(), data.size(), MADV_FLAGS);
#undef MADV_FLAGS
    return csc2sparse<FT>(matview, skip_empty);
}

template<typename FT=float, bool SO=blaze::rowMajor>
blz::SM<FT, SO> transposed_mtx2sparse(std::ifstream &ifs, size_t cols, size_t nr, size_t nnz) {
    blz::SM<FT, SO> ret(nr, cols);
    ret.reserve(nnz);
    std::vector<std::tuple<size_t, size_t, FT>> indices(nnz);
    size_t i = 0;
    for(std::string line;std::getline(ifs, line);) {
        size_t row, col;
        char *s = line.data();
        col = std::strtoull(line.data(), &s, 10) - 1;
        row = std::strtoull(s, &s, 10) - 1;
        FT cnt = std::atof(s);
        indices[i++] = {row, col, cnt};
        assert(row < nr || !std::fprintf(stderr, "ret rows: %zu. row ind %zu\n", ret.columns(), row));
        assert(col < cols || !std::fprintf(stderr, "ret columns: %zu. col ind %zu\n", ret.rows(), col));
    }
    shared::sort(indices.begin(), indices.end());
    size_t ci = 0;
    for(const auto [x, y, v]: indices) {
        while(ci < x) ret.finalize(ci++);
        ret.append(x, y, v);
    }
    while(ci < nr) ret.finalize(ci++);
    //std::fprintf(stderr, "ret has %zu columns, %zu rows\n", ret.columns(), ret.rows());
    transpose(ret);
    std::fprintf(stderr, "ret has %zu columns, %zu rows after transposition\n", ret.columns(), ret.rows());
    return ret;
}

template<typename FT=float, bool SO=blaze::rowMajor>
blz::SM<FT, SO> mtx2sparse(std::string prefix, bool perform_transpose=false)
{
    std::string line;
    std::ifstream ifs(prefix);
    do std::getline(ifs, line); while(line.front() == '%');
    char *s;
    size_t columns = std::strtoull(line.data(), &s, 10),
             nr    = std::strtoull(s, &s, 10),
             lines = std::strtoull(s, nullptr, 10);
    if(perform_transpose) {
        return transposed_mtx2sparse(ifs, columns, nr, lines);
    }
    blz::SM<FT, SO> ret(nr, columns);
    std::vector<std::pair<size_t, FT>> indices;
    size_t lastrow = 0;
    ret.reserve(lines);
    while(lines--)
    {
        if(!std::getline(ifs, line)) {
            const char *s = "Error in reading file: unexpected number of lines";
            std::cerr << s;
            throw std::runtime_error(s);
        }
        size_t row, col;
        col = std::strtoull(line.data(), &s, 10) - 1;
        row = std::strtoull(s, &s, 10) - 1;
        if(perform_transpose) std::swap(col, row);
        FT cnt = std::atof(s);
        if(row < lastrow) throw std::runtime_error("Unsorted file");
        else if(row != lastrow) {
            //std::fprintf(stderr, "lastrow %zu has %zu indices\n", row, indices.size());
            std::sort(indices.begin(), indices.end());
            for(const auto [idx, cnt]: indices)
                ret.append(lastrow, idx, cnt);
            indices.clear();
            while(lastrow < row) ret.finalize(lastrow++);
        }
        indices.emplace_back(col, cnt);
    }
    std::sort(indices.begin(), indices.end());
    for(const auto [idx, cnt]: indices) {
        ret.append(lastrow, idx, cnt);
    }
    while(lastrow < ret.rows()) ret.finalize(lastrow++);
    if(std::getline(ifs, line)) throw std::runtime_error("Error reading file: too many lines");
    std::fprintf(stderr, "Parsed file of %zu rows/%zu columns\n", ret.rows(), ret.columns());
    return ret;
}

template<typename FT=float, bool SO=blaze::rowMajor, typename IndPtrType, typename IndicesType, typename DataType>
blz::SM<FT, SO> csc2sparse(const IndPtrType *indptr, const IndicesType *indices, const DataType *data, 
                           size_t nnz, size_t nfeat, uint32_t nitems) {
    return csc2sparse<FT, SO>(CSCMatrixView<IndPtrType, IndicesType, DataType>(indptr, indices, data, nnz, nfeat, nitems));
}

} // namespace util
using util::csc2sparse;
using util::mtx2sparse;
using util::nonZeros;
using util::CSCMatrixView;
using util::is_csc_view;
using util::is_csc_view_v;

} // namespace minocore

#endif /* CSC_H__ */
