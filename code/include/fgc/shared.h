#pragma once
#include "robin-hood-hashing/src/include/robin_hood.h"
#include "aesctr/wy.h"
#include "macros.h"
#include "flat_hash_map/flat_hash_map.hpp"

#if defined(USE_TBB)
#include <execution>
#  define inclusive_scan(x, y, z) inclusive_scan(::std::execution::par_unseq, x, y, z)
#else
#  define inclusive_scan(x, y, z) ::std::partial_sum(x, y, z)
#endif

namespace shared {
template <typename Key, typename T, typename Hash = robin_hood::hash<Key>,
          typename KeyEqual = std::equal_to<Key>, size_t MaxLoadFactor100 = 80>
using flat_hash_map = robin_hood::unordered_flat_map<Key, T, Hash, KeyEqual, MaxLoadFactor100>;
template<typename T, typename H = std::hash<T>, typename E = std::equal_to<T>, typename A = std::allocator<T> >
using flat_hash_set = ska::flat_hash_set<T, H, E, A>;

}
