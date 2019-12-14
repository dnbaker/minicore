#pragma once
#include <cassert>
#include <map>
#include <mutex>
#include <vector>
#include "aesctr/wy.h"
#include "macros.h"

#if defined(USE_TBB)
#include <execution>
#  define inclusive_scan(x, y, z) inclusive_scan(::std::execution::par_unseq, x, y, z)
#else
#  define inclusive_scan(x, y, z) inclusive_scan(x, y, z)
//#  define inclusive_scan(x, y, z) ::std::partial_sum(x, y, z)
#endif

namespace clustering {
using std::inclusive_scan;
using std::partial_sum;

template<typename C>
using ContainedTypeFromIterator = std::decay_t<decltype((*std::declval<C>())[0])>;


template<typename Iter, typename FT=ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG>
std::vector<IT>
kmeanspp(Iter first, Iter end, RNG &rng, size_t k) {
    static_assert(std::is_arithmetic<FT>::value, "FT must be arithmetic");
    size_t np = std::distance(first, end);
    std::vector<IT> centers;
    std::vector<FT> distances(np, 0.), cdf(np);
    std::vector<IT> assignments(np, IT(-1));
    bool firstround = true;
    double sumd2 = 0.;
    {
        auto fc = rng() % np;
        centers.push_back(fc);
        auto &lhs = first[centers.front()];
#ifdef _OPENMP
        OMP_PRAGMA("omp parallel for reduction(+:sumd2)")
        for(size_t i = 0; i < np; ++i) {
            double dist = l2Dist(lhs, first[i]);
            distances[i] = dist;
            sumd2 += dist;
        }
#else
        SK_UNROLL_8
        for(size_t i = 0; i < np; ++i) {
            auto dist = l2Dist(lhs, first[i]);
            distances[i] = dist;
            sumd2 += dist;
        }
#endif
        assert(distances[fc] == 0.);
        inclusive_scan(distances.begin(), distances.end(), cdf.begin());
    }
#if VERBOSE_AF
    std::fprintf(stderr, "first loop sum: %f. manual: %f\n", sumd2, std::accumulate(distances.begin(), distances.end(), double(0)));
#endif
        
    // TODO: keep track of previous centers so that we don't re-compare
    // (using assignments vector)
    while(centers.size() < k) {
        // At this point, the cdf has been prepared, and we are ready to sample.
        // add new element
        auto newc = std::lower_bound(cdf.begin(), cdf.end(), cdf.back() * FT(rng()) / rng.max()) - cdf.begin();
        centers.push_back(newc);
        auto &lhs = first[newc];
        sumd2 -= distances[newc];
        distances[newc] = 0.;
        double sum = sumd2;
        OMP_PRAGMA("omp parallel for")
        for(IT i = 0; i < np; ++i) {
            auto &ldist = distances[i];
            //if(ldist == 0.) continue;
            auto dist = FT(l2Dist(lhs, first[i]));
            auto &lhs = first[i];
            if(dist < ldist) { // Only write if it changed
                auto diff = dist - ldist;
                #pragma omp atomic
                sum += diff;
                ldist = dist;
            }
        }
        sumd2 = sum;
#if VERBOSE_AF
        std::fprintf(stderr, "sumd2: %f. manual: %f\n", sumd2, std::accumulate(distances.begin(), distances.end(), double(0)));
#endif
        inclusive_scan(distances.begin(), distances.end(), cdf.begin());
    }
    return centers;
}

#undef inclusive_scan

} // clustering
