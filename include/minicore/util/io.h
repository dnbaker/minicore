#ifndef MINOCORE_UTIL_IO_H__
#define MINOCORE_UTIL_IO_H__
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <iostream>
#include <fstream>

#ifdef _BZLIB_H
  #include "bzlib.h"
  #include <boost/iostreams/filter/bzip2.hpp>
  #pragma message("Enabling bzip2 support")
  #ifndef EXTERNAL_BOOST_IOSTREAMS
    #include "./boost/bzip2.cpp"
  #else
  #endif
#else
  #ifdef VERBOSE_AF
    #pragma message("Not enabling bzip2 support")
  #endif
#endif

#if defined(LZMA_H) || defined(HAVE_LZMA)
  #include "lzma.h"
  #include <boost/iostreams/filter/lzma.hpp>
  #ifdef VERBOSE_AF
    #pragma message("Enabling xz support")
  #endif
  #ifndef EXTERNAL_BOOST_IOSTREAMS
    #include "./boost/lzma.cpp"
  #endif
#else
  #ifdef VERBOSE_AF
    #pragma message("Not Enabling xz support")
  #endif
#endif


#ifdef ZSTDLIB_API
  #include <boost/iostreams/filter/zstd.hpp>
  #ifdef VERBOSE_AF
    #pragma message("Enabling zstd support")
  #endif
  #ifndef EXTERNAL_BOOST_IOSTREAMS
    #include "./boost/zstd.cpp"
  #endif
#else
  #ifdef VERBOSE_AF
    #pragma message("Not Enabling zstsd support")
  #endif
#endif

#include <boost/algorithm/string/predicate.hpp>

#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/zlib.hpp>

#ifndef EXTERNAL_BOOST_IOSTREAMS
#include "./boost/gzip.cpp"
#include "./boost/zlib.cpp"
#endif


namespace minicore {

namespace util {

namespace io {

namespace boost_io = boost::iostreams;

static inline
std::pair<std::unique_ptr<boost_io::filtering_istream>, std::unique_ptr<std::ifstream>>
xopen(std::string path) {
    std::pair<std::unique_ptr<boost_io::filtering_istream>, std::unique_ptr<std::ifstream>> ret;
    ret.first.reset(new boost_io::filtering_istream);
    if(boost::algorithm::ends_with(path, ".gz")) {
        ret.first->push(boost_io::gzip_decompressor());
#ifdef LZMA_H
    } else if(boost::algorithm::ends_with(path, ".xz")) {
        ret.first->push(boost_io::lzma_decompressor());
#endif
#ifdef _BZLIB_H
    } else if(boost::algorithm::ends_with(path, ".bz2")) {
        ret.first->push(boost_io::bzip2_decompressor());
#endif
#ifdef ZSTDLIB_API
    } else if(boost::algorithm::ends_with(path, ".zst")) {
        ret.first->push(boost_io::basic_zstd_decompressor());
#endif
    }
    ret.second.reset(new std::ifstream(path,  std::ios_base::in | std::ios_base::binary));
    ret.first->push(*ret.second);
    return ret;
}

} // namespace io

}

} // namespace minicore

#endif /* MINOCORE_UTIL_IO_H__ */
