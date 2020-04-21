#ifndef CSC_H__
#define CSC_H__
#include "./shared.h"
#include "./timer.h"
#include "./blaze_adaptor.h"
#include "mio/single_include/mio/mio.hpp"
#include <fstream>

namespace minocore {

struct CSCMatrixView {
    const uint64_t *const indptr_, *const indices_;
    const uint32_t *const data_;
    const uint64_t nnz_;
    const uint32_t nf_, n_;
    CSCMatrixView(const uint64_t *indptr, const uint64_t *indices, const uint32_t *data,
                  uint64_t nnz, uint32_t nfeat, uint32_t nitems):
        indptr_(indptr),
        indices_(indices),
        data_(data),
        nnz_(nnz),
        nf_(nfeat), n_(nitems)
    {
    }
    struct Column {
        const CSCMatrixView &mat_;
        Column(const CSCMatrixView &mat, size_t start, size_t stop): mat_(mat), start_(start), stop_(stop) {}
        size_t start_;
        size_t stop_;
        size_t nnz() const {return stop_ - start_;}
    };
    auto column(size_t i) const {
        return Column(*this, indptr_[i], indptr_[i + 1]);
    }
};

template<typename FT=float>
blz::SM<FT, blaze::rowMajor> csc2sparse(const CSCMatrixView &mat, bool skip_empty=false) {
    blz::SM<FT, blaze::rowMajor> ret(mat.n_, mat.nf_);
    ret.reserve(mat.nnz_);
    size_t used_rows = 0, i;
    for(i = 0; i < mat.n_; ++i) {
        auto col = mat.column(i);
        if(mat.n_ > 1000000 && i % 1000000 == 0) std::fprintf(stderr, "%zu/%u\r", i, mat.n_);
        if(skip_empty && 0u == col.nnz()) continue;
        for(auto s = col.start_; s < col.stop_; ++s) {
            ret.append(used_rows, mat.indices_[s], mat.data_[s]);
        }
        ret.finalize(used_rows++);
    }
    if(used_rows != i) std::fprintf(stderr, "Only used %zu/%zu rows, skipping empty rows\n", used_rows, i);
    return ret;
}

template<typename FT=float>
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
    CSCMatrixView matview((const uint64_t *)indptr.data(), (const uint64_t *)indices.data(),
                          (const uint32_t *)data.data(), indices.size() / (sizeof(uint64_t) / sizeof(indices[0])),
                          nfeat, nsamples);
#ifndef MADV_REMOVE
#  define MADV_FLAGS (MADV_DONTNEED | MADV_FREE)
#else
#  define MADV_FLAGS (MADV_DONTNEED | MADV_REMOVE)
#endif
    ::madvise((void *)indptr.data(), indptr.size(), MADV_FLAGS);
    ::madvise((void *)indices.data(), indices.size(), MADV_FLAGS);
    ::madvise((void *)data.data(), data.size(), MADV_FLAGS);
#undef MADV_FLAGS
    return csc2sparse(matview, skip_empty);
}

template<typename FT=float, bool SO=blaze::rowMajor>
blz::SM<FT, SO> mtx2sparse(std::string prefix)
{
    std::string line;
    std::ifstream ifs(prefix);
    do std::getline(ifs, line); while(line.front() == '%');
    char *s;
    size_t columns = std::strtoull(line.data(), &s, 10),
             nr  = std::strtoull(s, &s, 10),
             lines = std::strtoull(s, nullptr, 10);
    size_t lastline = 0;
    blz::SM<FT, SO> ret(nr, columns);
    ret.reserve(lines);
    while(lines--)
    {
        if(!std::getline(ifs, line)) throw std::runtime_error("Error in reading file: unexpected number of lines");
        size_t row, col, cnt;
        col = std::strtoull(line.data(), &s, 10) - 1;
        row = std::strtoull(s, &s, 10) - 1;
        cnt = std::strtoull(s, &s, 10);
        if(unlikely(lastline > row)) throw std::runtime_error("Unsorted file.");
        while(lastline < row) ret.finalize(lastline++);
        ret.append(row, col, cnt);
    }
    while(lastline < ret.rows())
        ret.finalize(lastline++);
    if(std::getline(ifs, line)) throw std::runtime_error("Error reading file: too many lines");
    return ret;
}

} // namespace minocore

#endif /* CSC_H__ */
