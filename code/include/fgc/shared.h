#pragma once
#include "robin-hood-hashing/src/include/robin_hood.h"
#include "aesctr/wy.h"
#include "macros.h"
#include "flat_hash_map/flat_hash_map.hpp"
#include <system_error>
#include <cassert>
#include <unistd.h>

#if defined(USE_TBB)
#include <execution>
#  define inclusive_scan(x, y, z) inclusive_scan(::std::execution::par_unseq, x, y, z)
#else
#  define inclusive_scan(x, y, z) ::std::partial_sum(x, y, z)
#endif

namespace fgc {


#if !NDEBUG
template<typename T> class TD; // Debugging only
#endif


namespace shared {
template <typename Key, typename T, typename Hash = robin_hood::hash<Key>,
          typename KeyEqual = std::equal_to<Key>, size_t MaxLoadFactor100 = 80>
using flat_hash_map = robin_hood::unordered_flat_map<Key, T, Hash, KeyEqual, MaxLoadFactor100>;
template<typename T, typename H = std::hash<T>, typename E = std::equal_to<T>, typename A = std::allocator<T> >
using flat_hash_set = ska::flat_hash_set<T, H, E, A>;

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

} // shared
} // fgc
