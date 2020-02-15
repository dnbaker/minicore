#ifndef CSC_H__
#define CSC_H__
#include "fgc/shared.h"
#include "fgc/blaze_adaptor.h"
#include "mio/single_include/mio/mio.hpp"
namespace fgc {

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
    size_t used_rows = 0;
    for(unsigned i = 0; i < mat.n_; ++i) {
        auto col = mat.column(i);
        if(skip_empty && 0u == col.nnz()) continue;
        for(auto s = col.start_; s < col.stop_; ++s) {
            ret.append(used_rows, mat.indices_[s], mat.data_[s]);
        }
        ret.finalize(used_rows++);
    }
    return ret;
}

template<typename FT=float>
blz::SM<FT, blaze::rowMajor> csc2sparse(std::string prefix, bool skip_empty=false) {
    std::string indptrn  = prefix + "indptr.file";
    std::string indicesn = prefix + "indices.file";
    std::string datan    = prefix + "data.file";
    std::string shape    = prefix + "shape.file";
    std::FILE *ifp = std::fopen(shape.data(), "rb");
    uint32_t dims[2];
    std::fread(dims, sizeof(uint32_t), 2, ifp);
    uint32_t nfeat = dims[0], nsamples = dims[1];
    std::fprintf(stderr, "nfeat: %u. nsample: %u\n", nfeat, nsamples);
    std::fclose(ifp);
    using mmapper = mio::mmap_source;
    mmapper indptr(indptrn), indices(indicesn), data(datan);
    CSCMatrixView matview((const uint64_t *)indptr.data(), (const uint64_t *)indices.data(),
                          (const uint32_t *)data.data(), indices.size() / (sizeof(uint64_t) / sizeof(indices[0])),
                          nfeat, nsamples);
    return csc2sparse(matview, skip_empty);
}


}

#endif /* CSC_H__ */
