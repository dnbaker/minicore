#pragma once
#include "unordered_dense/include/ankerl/unordered_dense.h"
#include "thirdparty/kxsort.h"
#include "thirdparty/pdqsort.h"
#include "aesctr/wy.h"
#include "./macros.h"
#include <system_error>
#include <cassert>
#include <unistd.h>


#ifndef BOOST_NO_AUTO_PTR
#define BOOST_NO_AUTO_PTR 1
#endif

#if defined(USE_TBB)
#  include <execution>
#endif

namespace minicore {

#if !NDEBUG
template<typename T> class TD; // Debugging only
#endif

#ifndef FGC_MAX_HASH_LOAD_FACTOR
#define FGC_MAX_HASH_LOAD_FACTOR 80
#endif


namespace shared {
template <typename Key, typename T, typename Hash = ankerl::unordered_dense::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
using flat_hash_map = ankerl::unordered_dense::map<Key, T, Hash, KeyEqual>;

template<typename T, typename H = ankerl::unordered_dense::hash<T>, typename E = std::equal_to<T>>
using flat_hash_set = ankerl::unordered_dense::set<T, H, E>;

using ssize_t = std::ptrdiff_t;

INLINE auto checked_posix_write(int fd, const void *buf, ssize_t count) {
    ssize_t ret = ::write(fd, buf, count);
    if(unlikely(ret != count)) {
        if(ret == -1) {
            throw std::system_error(errno, std::system_category(), "Failed to ::write entirely");
        }
        assert(ret < count);
        char buf[256];
        std::sprintf(buf, "Failed to write fully. (Wrote %zd bytes instead of %zd)", ret, count);
        throw std::system_error(0, std::system_category(), buf);
    }
    return ret;
}


struct Deleter {
    void operator()(const void *x) const {
        std::free(const_cast<void *>(x));
    }
};

template<typename It, typename Cmp=std::less<>,
         typename=std::enable_if_t<!std::is_integral_v<std::decay_t<decltype(*std::declval<It>())>>
                                   || !std::is_same_v<Cmp, std::less<>>>
        >
INLINE void sort(It beg, It end, Cmp cmp=Cmp()) {
    pdqsort(beg, end, cmp);
}

template<typename It,
         typename=std::enable_if_t<std::is_integral_v<std::decay_t<decltype(*std::declval<It>())>>>>
INLINE void sort(It beg, It end) {
    kx::radix_sort(beg, end);
}

template<typename T>
struct dumbrange {
    T beg, e_;
    dumbrange(T beg, T end): beg(beg), e_(end) {}
    auto begin() const {return beg;}
    auto end()   const {return e_;}
};

template<typename T>
inline dumbrange<T> make_dumbrange(T beg, T end) {return dumbrange<T>(beg, end);}

template<typename C>
using ContainedTypeFromIterator = std::decay_t<decltype((*std::declval<C>())[0])>;

} // shared
using shared::ssize_t;
} // minicore
