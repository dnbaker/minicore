#pragma once
#include <vector>
#include <map>
#include <cassert>
#include <atomic>
#include <mutex>
#include "aesctr/wy.h"
#include "macros.h"

namespace clustering {

template<typename C>
using ContainedTypeFromIterator = std::decay_t<decltype((*std::declval<C>())[0])>;

template<typename FT, typename A, typename OA>
double sqrNorm(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    double s = 0.;
    for(size_t i = 0; i < lhs.size(); ++i) {
        FT tmp = lhs[i] - rhs[i];
        s += tmp * tmp;
    }
    return s;
}


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
            double dist = sqrNorm(lhs, first[i]);
            distances[i] = dist;
            sumd2 += dist;
        }
#else
        SK_UNROLL_8
        for(size_t i = 0; i < np; ++i) {
            auto dist = sqrNorm(lhs, first[i]);
            distances[i] = dist;
            sumd2 += dist;
        }
#endif
        assert(distances[fc] == 0.);
#ifdef _OPENMP
        OMP_PRAGMA("omp parallel for")
#else
        SK_UNROLL_8
#endif
        for(IT i = 0; i < np; ++i)
            cdf[i] = distances[i] / sumd2;
        std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());
        cdf.back() = 1.;
    }
    std::fprintf(stderr, "first loop sum: %f. manual: %f\n", sumd2, std::accumulate(distances.begin(), distances.end(), FT(0)));
        
    // TODO: keep track of previous centers so that we don't re-compare
    // (using assignments vector)
    while(centers.size() < k) {
        // At this point, the cdf has been prepared, and we are ready to sample.
        // add new element
        auto newc = std::lower_bound(cdf.begin(), cdf.end(), FT(rng()) / rng.max()) - cdf.begin();
        centers.push_back(newc);
        auto &lhs = first[newc];
        sumd2 -= distances[newc];
        distances[newc] = 0.;
        double sum = sumd2;
        OMP_PRAGMA("omp parallel for")
        for(IT i = 0; i < np; ++i) {
            auto &ldist = distances[i];
            //if(ldist == 0.) continue;
            auto dist = FT(sqrNorm(lhs, first[i]));
            auto &lhs = first[i];
            if(dist < ldist) { // Only write if it changed
                auto diff = dist - ldist;
                #pragma omp atomic
                sum += diff;
                ldist = dist;
            }
        }
        sumd2 = sum;
        std::fprintf(stderr, "sumd2: %f. manual: %f\n", sumd2, std::accumulate(distances.begin(), distances.end(), FT(0)));
        const auto sd2i = 1. / sumd2;
#ifdef _OPENMP
        OMP_PRAGMA("omp parallel for")
#else
        SK_UNROLL_8
#endif
        for(IT i = 0; i < np; ++i)
            cdf[i] = distances[i] * sd2i; // Maybe SIMD later?
        std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());
        cdf.back() = 1.;
    }
    return centers;
}

} // clustering
