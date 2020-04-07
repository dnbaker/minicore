#pragma once
#include "robin-hood-hashing/src/include/robin_hood.h"
#include "pdqsort/pdqsort.h"
#include "aesctr/wy.h"
#include "macros.h"
#include <system_error>
#include <cassert>
#include <unistd.h>

#if defined(USE_TBB)
#  include <execution>
#  define inclusive_scan(...) inclusive_scan(::std::execution::par_unseq, __VA_ARGS__)
#  ifndef NDEBUG
#    pragma message("Using parallel inclusive scan")
#  endif
#else
#  define inclusive_scan(...) ::std::partial_sum(__VA_ARGS__)
#endif

namespace fgc {

#ifdef USE_TBB
using std::inclusive_scan;
#endif

#if !NDEBUG
template<typename T> class TD; // Debugging only
#endif


namespace shared {
template <typename Key, typename T, typename Hash = robin_hood::hash<Key>,
          typename KeyEqual = std::equal_to<Key>, size_t MaxLoadFactor100 = 80>
using flat_hash_map = robin_hood::unordered_flat_map<Key, T, Hash, KeyEqual, MaxLoadFactor100>;

template<typename T, typename H = robin_hood::hash<T>, typename E = std::equal_to<T>>
using flat_hash_set = robin_hood::unordered_flat_set<T, H, E>;

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

template<typename It, typename Cmp=std::less<>>
INLINE void sort(It beg, It end, Cmp cmp=Cmp()) {
    pdqsort(beg, end, cmp);
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

template<typename T1, typename T2>
struct packed_pair {
    T1 first;
    T2 second;
    packed_pair(T1 f, T2 sec): first(f), second(sec) {}
    packed_pair(): first(), second() {}
    template<class _U1, class _U2>
         packed_pair(_U1&& __x, _U2&& __y)
     : first(std::forward<_U1>(__x)),
       second(std::forward<_U2>(__y)) { }
    packed_pair(packed_pair &&o)     = default;
    packed_pair(const packed_pair &o) = default;
    template<class _U1, class _U2>
        packed_pair(packed_pair<_U1, _U2>&& __p): first(std::move(__p.first)),
                                                  second(std::move(__p.second)) { }
    template<typename V>
    packed_pair(std::initializer_list<V> l): first(std::move(l[0])), second(std::move(l[1])) {}
    packed_pair &operator=(const packed_pair &o) {
        first = o.first; second = o.second; return *this;
    }
    bool operator<(const packed_pair &o) const {return std::tie(first, second) < std::tie(o.first, o.second);}
    bool operator>(const packed_pair &o) const {return std::tie(first, second) > std::tie(o.first, o.second);}
    bool operator<=(const packed_pair &o) const {return std::tie(first, second) <= std::tie(o.first, o.second);}
    bool operator>=(const packed_pair &o) const {return std::tie(first, second) >= std::tie(o.first, o.second);}
    bool operator==(const packed_pair &o) const {return std::tie(first, second) == std::tie(o.first, o.second);}
    bool operator!=(const packed_pair &o) const {return std::tie(first, second) != std::tie(o.first, o.second);}
}
#if __GNUC__ || __clang__
__attribute__((__packed__))
#endif
;

} // shared
} // fgc
