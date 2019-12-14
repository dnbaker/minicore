#pragma once
#include <vector>
#include <map>
#include <atomic>
#include "aesctr/wy.h"
#include "macros.h"

namespace clustering {

template<typename C>
using ContainedTypeFromIterator = std::decay_t<decltype((*std::declval<C>())[0])>;

template<typename FT, typename A>
double sqrNorm(const std::vector<FT, A> &lhs, const std::vector<FT, A> &rhs) {
    double s = 0.;
    for(size_t i = 0; i < lhs.size(); ++i)
        s += std::pow(lhs[i] - rhs[i], 2);
    return s;
}

template<typename Iter, typename FT=ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG>
std::vector<IT>
kmeanspp(Iter first, Iter end, RNG &rng, size_t k) {
    static_assert(std::is_arithmetic<FT>::value, "FT must be arithmetic");
    size_t np = std::distance(first, end);
    std::vector<IT> centers;
    std::vector<float> distances(np, std::numeric_limits<float>::max()), cdf(np);
    std::vector<IT> assignments(np, IT(-1));
    bool firstround = true;
    FT sumd2 = 0.;
    {
        centers.push_back(rng() % np);
        auto &lhs = first[centers.front()];
        std::atomic<FT>  sum;
        sum.store(FT(0.));
        OMP_PRAGMA("omp parallel for")
        for(size_t i = 0; i < np; ++i) {
            auto dist = i == centers.front() ? FT(0.): FT(sqrNorm(lhs, first[i]));
            distances[i] = dist;
            FT current = sum.load();
            while(!sum.compare_exchange_weak(current, current + dist));
        }
        sumd2 = sum;
        auto sd2i = 1. / sumd2;
        SK_UNROLL_8
        OMP_PRAGMA("omp parallel for")
        for(IT i = 0; i < np; ++i)
            cdf[i] = distances[i] * sd2i; // Maybe SIMD later?
        std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());
        cdf.back() = 1.;
    }
    // TODO: keep track of previous centers so that we don't re-compare
    // (using assignments vector)
    while(centers.size() < k) {
        // At this point, the cdf has been prepared, and we are ready to sample.
        // add new element
        auto newc = std::lower_bound(cdf.begin(), cdf.end(), double(rng()) / rng.max()) - cdf.begin();
        centers.push_back(newc);
        auto &lhs = first[newc];
        distances[newc] = 0.;
        for(IT i = 0; i < np; ++i) {
            auto &ldist = distances[i];
            if(ldist == 0.) continue;
            auto dist = FT(sqrNorm(lhs, first[i]));
            auto &lhs = first[i];
            if(dist < ldist) { // Only write if it changed
                sumd2 += dist - ldist;
                ldist = dist;
            }
        }
        auto sd2i = 1. / sumd2;
        //OMP_PRAGMA("omp parallel for")
        OMP_PRAGMA("omp parallel for")
        SK_UNROLL_8
        for(IT i = 0; i < np; ++i)
            cdf[i] = distances[i] * sd2i; // Maybe SIMD later?
        std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());
        cdf.back() = 1.;
    }
    return centers;
}

} // clustering
