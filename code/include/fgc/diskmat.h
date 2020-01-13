#pragma once
#ifndef DISK_MAT_H__
#define DISK_MAT_H__
#include <memory>
#include "mio/single_include/mio/mio.hpp"
#include <cstring>
#include "blaze/Math.h"
namespace fgc {

template<typename VT, bool SO=blaze::rowMajor, bool isPadded=blaze::padded, bool isAligned=blaze::unaligned>
struct DiskMat {
    size_t nr_, nc_;
    using mmapper = mio::mmap_sink;
    std::unique_ptr<mmapper> ms_;
    bool delete_file_;


    // TODO:
    //  alignment -- if offset is 0, it's already aligned.
    //            -- otherwise, allocate enough extra so that it is

    static constexpr bool AF = isAligned;
    static constexpr bool PF = isPadded;
    using MatType = blaze::CustomMatrix<VT, AF, PF, SO>;
    MatType mat_;
    std::string path_;
    DiskMat(const DiskMat &o) = delete;
    DiskMat(DiskMat &&o) {
        uint8_t *ptr = reinterpret_cast<uint8_t *>(this), *optr = reinterpret_cast<uint8_t *>(std::addressof(o));
        std::memset(ptr, 0, sizeof(*this));
        std::swap_ranges(ptr, ptr + sizeof(*this), optr);
    }
    static constexpr size_t SIMDSIZE = blaze::SIMDTrait<VT>::size;
    DiskMat(size_t nr, size_t nc, std::string path, size_t offset=0, bool delete_file=false):
        nr_(nr), nc_(nc),
        delete_file_(delete_file),
        path_(path)
    {
        if(isAligned && offset % (SIMDSIZE * sizeof(VT))) {
            throw std::invalid_argument("offset is not aligned; invalid storage.");
        }
        const size_t nperrow = isPadded ? size_t(blaze::nextMultiple(nc_, SIMDSIZE)): nc_;
        const size_t nb = nr_ * nperrow * sizeof(VT), total_nb = nb + offset;
        const char *fn = path.data();
        std::FILE *fp = fopen(fn, "a+");
        if(!fp) throw 1;
        auto fd = ::fileno(fp);
        struct stat st;
        auto rc = ::fstat(fd, &st);
        if(rc) {
            char buf[256];
            std::sprintf(buf, "Failed to fstat fd/fp/path %d/%p/%s", fd, (void *)fp, path.data());
            std::fclose(fp);
            throw std::system_error(rc, std::system_category(), buf);
        }
        size_t filesize = st.st_size;
        if(filesize < total_nb) {
            if((rc = ::ftruncate(fd, total_nb))) throw std::system_error(rc, std::system_category(), "Failed to resize (ftruncate)");
            ::fstat(fd, &st);
        }
        assert(size_t(st.st_size) >= total_nb);
        ms_.reset(new mmapper(fd, offset, nb));
        mat_ = MatType((VT *)ms_->data(), nr, nc, nperrow);
    }
    auto operator()(size_t i, size_t j) const {return (~*this)(i, j);}
    auto &operator()(size_t i, size_t j)      {return (~*this)(i, j);}
    ~DiskMat() {
        if(delete_file_) {
            auto rc = std::system((std::string("rm ") + path_).data());
            if(rc) {
                auto exit_status = WEXITSTATUS(rc);
                auto stopsig = WSTOPSIG(rc);
                throw std::system_error(exit_status, std::error_category(),
                                        std::string("File deletion failed with exit status ") + std::to_string(exit_status) + (stopsig ? std::to_string(stopsig): std::string()));
            }
        }
    }
    MatType       &operator~()       {return mat_;}
    const MatType &operator~() const {return mat_;}
}; // DiskMat

template<typename VT, bool SO, bool isPadded, bool isAligned, bool checked=true>
auto row(DiskMat<VT, SO, isPadded, isAligned> &mat, size_t i, blaze::Check<checked> check=blaze::Check<checked>()) {
    return blaze::row(~mat, i, check);
}
template<typename VT, bool SO, bool isPadded, bool isAligned, bool checked=true>
auto column(DiskMat<VT, SO, isPadded, isAligned> &mat, size_t i, blaze::Check<checked> check=blaze::Check<checked>()) {
    return blaze::row(~mat, i, check);
}

} // fgc

#endif
