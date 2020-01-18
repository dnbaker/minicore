#pragma once
#ifndef DISK_MAT_H__
#define DISK_MAT_H__
#include <memory>
#include "mio/single_include/mio/mio.hpp"
#include <cstring>
#include "blaze/Math.h"
#include <system_error>
namespace fgc {

template<typename VT, bool SO=blaze::rowMajor, bool isPadded=blaze::padded, bool isAligned=blaze::unaligned>
struct DiskMat {
    size_t nr_, nc_;
    using mmapper = mio::mmap_sink;
    std::unique_ptr<mmapper> ms_;
    std::FILE *fp_;
    bool delete_file_;


    // TODO:
    //  alignment -- if offset is 0, it's already aligned.
    //            -- otherwise, allocate enough extra so that it is

    static constexpr bool AF = isAligned;
    static constexpr bool PF = isPadded;
    using MatType = blaze::CustomMatrix<VT, AF, PF, SO>;
    MatType mat_;
    const char *path_;
    DiskMat(const DiskMat &o) = delete;
    DiskMat(DiskMat &&o) {
        uint8_t *ptr = reinterpret_cast<uint8_t *>(this), *optr = reinterpret_cast<uint8_t *>(std::addressof(o));
        std::memset(ptr, 0, sizeof(*this));
        std::swap_ranges(ptr, ptr + sizeof(*this), optr);
    }
    static constexpr size_t SIMDSIZE = blaze::SIMDTrait<VT>::size;
    DiskMat(const DiskMat &o, const char *s=nullptr, size_t offset=0, int delete_file=-1): DiskMat(o.rows(), o.columns(), s, offset, delete_file >= 0 ? delete_file: o.delete_file_)
    {
        std::memcpy(ms_->data(), o.ms_->data(), sizeof(VT) * (~*this).spacing() * nr_);
    }
    DiskMat(size_t nr, size_t nc, const char *s=nullptr, size_t offset=0, bool delete_file=false):
        nr_(nr), nc_(nc),
        delete_file_(delete_file),
        path_(s)
    {
        if(isAligned && offset % (SIMDSIZE * sizeof(VT))) {
            throw std::invalid_argument("offset is not aligned; invalid storage.");
        }
        const size_t nperrow = isPadded ? size_t(blaze::nextMultiple(nc_, SIMDSIZE)): nc_;
        const size_t nb = nr_ * nperrow * sizeof(VT), total_nb = nb + offset;
        if((fp_ = s ? std::fopen(s, "a+"): ::tmpfile()) == nullptr) {
            char buf[256];
            std::sprintf(buf, "Failed to open file for writing. %s/%d (%s)", ::strerror(errno), errno, s ? s: "tmpfil");
            throw std::system_error(0, std::system_category(), buf);
        }
        const int fd = ::fileno(fp_);
        struct stat st;
        int rc;
        if((rc = ::fstat(fd, &st))) {
            char buf[256];
            std::sprintf(buf, "Failed to fstat fd/fp/path %d/%p/%s", fd, (void *)fp_, path.data());
            std::fclose(fp_);
            fp_ = nullptr;
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
        if(fp_) std::fclose(fp_);
        if(delete_file_ && path_) {
            auto rc = std::system((std::string("rm ") + path_).data());
            if(rc) {
                std::fprintf(stderr, "Note: file deletion failed with exit status %d and stopsig %d\n",
                                      WEXITSTATUS(rc), WSTOPSIG(rc));
            }
        }
    }
    auto rows() const {return (~*this).rows();}
    auto columns() const {return (~*this).columns();}
    MatType       &operator~()       {return mat_;}
    const MatType &operator~() const {return mat_;}
}; // DiskMat

template<typename VT, bool SO, bool isPadded, bool isAligned, bool checked=true>
auto row(DiskMat<VT, SO, isPadded, isAligned> &mat, size_t i, blaze::Check<checked> check=blaze::Check<checked>()) {
    return blaze::row(~mat, i, check);
}
template<typename VT, bool SO, bool isPadded, bool isAligned, bool checked=true>
auto column(DiskMat<VT, SO, isPadded, isAligned> &mat, size_t i, blaze::Check<checked> check=blaze::Check<checked>()) {
    return blaze::column(~mat, i, check);
}

template<typename VT>
class DiskVector {
    size_t n_, m_;
    VT     *data_;
public:
    DiskVector(): n_(0), m_(0), n_(nullptr) {}
    size_t capacity() const {return m_;}
    size_t size()     const {return n_;}
    auto begin()       {return data_;}
    auto begin() const {return data_;}
    auto end()         {return data_ + n_;}
    auto end()   const {return data_ + n_;}
};

} // fgc

#endif
