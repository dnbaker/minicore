#pragma once
#ifndef DISK_MAT_H__
#define DISK_MAT_H__
#include <memory>
#include "mio/single_include/mio/mio.hpp"
#include <cstring>
#include "blaze/Math.h"
#include <system_error>
namespace fgc {

template<typename VT, bool SO=blaze::rowMajor, bool isPadded=blaze::padded, bool isAligned=blaze::aligned>
struct DiskMat {
    using This = DiskMat<VT, SO, isPadded, isAligned>;
    size_t nr_, nc_;
    using mmapper = mio::mmap_sink;
    std::unique_ptr<mmapper> ms_;
    std::FILE *fp_;
    bool delete_file_;


    // TODO:
    //  alignment -- if offset is 0, it's already aligned.
    //            -- otherwise, allocate enough extra so that it is

    static constexpr blaze::AlignmentFlag AF = isAligned ? blaze::aligned: blaze::unaligned;
    static constexpr blaze::PaddingFlag PF = isPadded ? blaze::padded: blaze::unpadded;
    using MatType = blaze::CustomMatrix<VT, AF, PF, SO>;
    MatType mat_;
    std::string path_;

    DiskMat(const DiskMat &o): DiskMat(o.nr_, o.nc_, nullptr) {
        std::memcpy(ms_->data(), o.ms_->data(), o.ms_->size());
    }
    DiskMat(DiskMat &&o): path_(o.path_) {
        uint8_t *ptr = reinterpret_cast<uint8_t *>(this), *optr = reinterpret_cast<uint8_t *>(std::addressof(o));
        std::memset(ptr, 0, sizeof(*this));
        std::swap_ranges(ptr, ptr + sizeof(*this), optr);
        std::fprintf(stderr, "[%s at %p] moved diskmat has path %s\n", __PRETTY_FUNCTION__, (void *)this, path_.data() ? path_.data(): "tmpfile");
    }
    static constexpr size_t SIMDSIZE = blaze::SIMDTrait<VT>::size;
    DiskMat(const DiskMat &o, const char *s, size_t offset=0, int delete_file=-1):
        DiskMat(o.rows(), o.columns(), s, offset, delete_file >= 0 ? delete_file: o.delete_file_)
    {
        std::memcpy(ms_->data(), o.ms_->data(), sizeof(VT) * (~*this).spacing() * nr_);
#if VERBOSE_AF
        std::fprintf(stderr, "Copied to %s\n", path_.size() ? path_.data(): "tmpfile");
#endif
    }
    operator       MatType &()       {return ~*this;}
    operator const MatType &() const {return ~*this;}
    DiskMat(size_t nr, size_t nc, const char *s=nullptr, size_t offset=0, bool delete_file=true):
        nr_(nr), nc_(nc),
        delete_file_(delete_file),
        path_(s ? s: "")
    {
#if VERBOSE_AF
        std::fprintf(stderr, "Opened file at %s to make matrix of size %zu, %zu\n", s ? s: "tmpfile", nr_, nc_);
#endif
        if(isAligned && offset % (SIMDSIZE * sizeof(VT))) {
            throw std::invalid_argument("offset is not aligned; invalid storage.");
        }
        const size_t nperrow = isPadded ? size_t(blaze::nextMultiple(nc_, SIMDSIZE)): nc_;
        const size_t nb = nr_ * nperrow * sizeof(VT), total_nb = nb + offset;
        if((fp_ = s ? std::fopen(s, "a+"): std::tmpfile()) == nullptr) {
            char buf[256];
            std::sprintf(buf, "Failed to open file for writing. %s/%d (%s)", ::strerror(errno), errno, s ? s: "tmpfil");
            throw std::system_error(0, std::system_category(), buf);
        }
        const int fd = ::fileno(fp_);
        struct stat st;
        int rc;
        if((rc = ::fstat(fd, &st))) {
            char buf[256];
            std::sprintf(buf, "Failed to fstat fd/fp/path %d/%p/%s", fd, (void *)fp_, path_.data());
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
        assert(s ? (path_.data() && std::strcmp(path_.data(), s)) == 0: path_.empty());
        std::fprintf(stderr, "Spacing: %zu\n", (~*this).spacing());
    }
    DiskMat(size_t nr, size_t nc, std::string path, size_t offset=0, bool delete_file=false): DiskMat(nr, nc, path.data(), offset, delete_file) {}
    auto operator()(size_t i, size_t j) const {return (~*this)(i, j);}
    auto &operator()(size_t i, size_t j)      {return (~*this)(i, j);}
    ~DiskMat() {
        if(fp_) std::fclose(fp_);
        if(delete_file_ && path_.size()) {
#if VERBOSE_AF
            std::fprintf(stderr, "[%s at %p]path: %s/%p\n", __PRETTY_FUNCTION__, (void *)this, path_.data(), (void *)path_.data());
#endif
            auto rc = std::system((std::string("rm ") + path_).data());
            if(rc) {
                std::fprintf(stderr, "Note: file deletion failed with exit status %d and stopsig %d\n",
                                      WEXITSTATUS(rc), WSTOPSIG(rc));
            }
        }
    }
    auto data() const {return mat_.data();}
    auto data()       {return mat_.data();}
    auto spacing() const {return mat_.spacing();}
    auto rows() const {return mat_.rows();}
    auto columns() const {return mat_.columns();}
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

#ifndef DEFAULT_MAX_NRAMBYTES
#define DEFAULT_MAX_NRAMBYTES static_cast<size_t>(16ull << 30)
#endif

template<typename VT, bool SO=blaze::rowMajor, size_t max_nbytes=DEFAULT_MAX_NRAMBYTES>
class PolymorphicMat {
    using CMType = blaze::CustomMatrix<VT, blaze::aligned, blaze::padded, blaze::rowMajor>;
    using DiskType = DiskMat<VT, SO, blaze::aligned, blaze::padded>;
    std::unique_ptr<DiskType> diskmat_;
    std::unique_ptr<blaze::DynamicMatrix<VT, SO>> rammat_;
    CMType cm_;
public:
    static constexpr size_t MAX_BYTES_RAM = max_nbytes;
    PolymorphicMat(size_t nr, size_t nc, size_t maxmem=MAX_BYTES_RAM, const char *s=nullptr) {
        size_t spacing = blaze::nextMultiple(nc, blaze::SIMDTrait<VT>::size);
        size_t total_bytes = nr * spacing * sizeof(VT);
        VT *ptr;
        if(total_bytes > maxmem) {
            diskmat_.reset(new DiskType(nr, nc, s));
            ptr = diskmat_->data();
        } else {
            rammat_.reset(new blaze::DynamicMatrix<VT, SO>(nr, nc));
            ptr = rammat_->data();
        }
        cm_ = CMType(ptr, nr, nc, spacing);
    }
    CMType &operator~() {return cm_;}
    const CMType &operator~() const {return cm_;}
};

} // fgc

#endif
