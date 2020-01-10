#pragma once
#ifndef DISK_MAT_H__
#define DISK_MAT_H__
#include <memory>
#include "mio/single_include/mio/mio.hpp"
#include <cstring>
#include "blaze/Math.h"
namespace fgc {

template<typename VT, bool SO=blaze::rowMajor>
struct DiskMat {
    size_t nr_, nc_;
    using mmapper = mio::mmap_sink;
    std::unique_ptr<mmapper> ms_;
    bool delete_file_ = false;
    // TODO:
    //  alignment -- if offset is 0, it's already aligned.
    //            -- otherwise, allocate enough extra so that it is
    //  padding   -- more importantly (to make later lines aligned)
    //            -- I should allocate a little extra space for full optimization

    static constexpr bool AF = blaze::unaligned;
    static constexpr bool PF = blaze::unpadded;
    using MatType = blaze::CustomMatrix<VT, AF, PF, SO>;
    MatType mat_;
    std::string path_;
    DiskMat(const DiskMat &o) = delete;
    DiskMat(DiskMat &&o) {
        uint8_t *ptr = reinterpret_cast<uint8_t *>(this), *optr = reinterpret_cast<uint8_t *>(std::addressof(o));
        std::memset(ptr, 0, sizeof(*this));
        std::swap_ranges(ptr, ptr + sizeof(*this), optr);
    }
    DiskMat(size_t nr, size_t nc, std::string path, size_t offset=0): nr_(nr), nc_(nc), path_(path) {
        const size_t nb = nr_ * nc_ * sizeof(VT), total_nb = nb + offset;
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
        auto filesize = st.st_size;
        if(size_t(filesize) < total_nb) {
            static constexpr size_t chunk_size = 1u << 16;
            std::vector<char> td(chunk_size, 0);
            for(size_t i = (total_nb - filesize) / chunk_size; i--;)
                ::write(fd, td.data(), chunk_size);
            ::fstat(fd, &st);
            if(total_nb != size_t(st.st_size)) {
                ::write(fd, td.data(), total_nb - st.st_size);
                ::fstat(fd, &st);
                assert(size_t(st.st_size) == total_nb);
            }
        }
        assert(size_t(st.st_size) >= total_nb);
        ms_.reset(new mmapper(fd, offset, nb));
        mat_ = MatType((VT *)ms_->data(), nr, nc);
    }
    ~DiskMat() {
        if(delete_file_) {
            std::system((std::string("rm ") + path_).data());
        }
    }
    MatType       &operator~()       {return mat_;}
    const MatType &operator~() const {return mat_;}
}; // DiskMat

}

#endif
