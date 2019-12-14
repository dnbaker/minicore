#pragma once
#include "kmeans.h"

#if defined(USE_TBB)
#include <execution>
#  define inclusive_scan(x, y, z) inclusive_scan(::std::execution::par_unseq, x, y, z)
#else
#  define inclusive_scan(x, y, z) ::std::partial_sum(x, y, z)
#endif

namespace clustering {
using std::inclusive_scan;
using std::partial_sum;
using blz::L2Norm;

// Replace 

template<typename Iter, typename FT=ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG, typename Norm=L2Norm>
std::vector<IT>
fp_kcenter(Iter first, Iter end, RNG &rng, size_t k, const Norm &norm=Norm()) {
    static_assert(std::is_arithmetic<FT>::value, "FT must be arithmetic");
    size_t np = std::distance(first, end);
    std::vector<IT> centers;
    std::vector<FT> distances(np, 0.), cdf(np);
    {
        auto fc = rng() % np;
        centers.push_back(fc);
        auto &lhs = first[centers.front()];
#ifdef _OPENMP
        OMP_PRAGMA("omp parallel for")
        for(size_t i = 0; i < np; ++i) {
            distances[i] = norm(lhs, first[i]);
        }
#else
        SK_UNROLL_8
        for(size_t i = 0; i < np; ++i) {
            distances[i] = norm(lhs, first[i]);
        }
#endif
        assert(distances[fc] == 0.);
    }
        
    while(centers.size() < k) {
        // At this point, the cdf has been prepared, and we are ready to sample.
        // add new element
        auto newc = std::max_element(distances.begin(), distances.end()) - distances.begin();
        centers.push_back(newc);
        auto &lhs = first[newc];
        OMP_PRAGMA("omp parallel for")
        for(IT i = 0; i < np; ++i) {
            auto &ldist = distances[i];
            if(auto dist(norm(lhs, first[i])); ldist < dist) ldist = dist;
        }
    }
    return centers;
}

#undef inclusive_scan

} // clustering
