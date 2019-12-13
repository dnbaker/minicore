#pragma once
#include <vector>
#include <map>
#include "aesctr/wy.h"

namespace clustering {

template<typename C>
using ContainedTypeFromIterator = std::decay_t<decltype(std::declval<C>()[0])>;

template<typename Iter, typename FT=ContainedTypeFromIterator,
         typename IT=std::uint32_t, typename RNG>
std::vector<IT>
kmeanspp(Iter first, Iter end, std::vector<FT> &norms, RNG &rng, size_t k) {
    static_assert(std::is_arithmetic<FT>::value, "FT must be arithmetic");
    norms.resize(std::distance(first, end));
    size_t np = norms.size();
    auto first = rng() % np;
    std::vector<IT> centers{first};
    std::vector<float> distances(np, std::numeric_limits<float>::max()), cdf(np);
    bool firstround = true;
    FT sumd2 = 0.;
    // TODO: keep track of previous centers so that we don't re-compare
    while(ret.size() < k) {
        for(IT i = 0; i < np; ++i) {
            for(const auto c: centers) {
                auto dist = (c == i) ? FT(0.): FT(sqrNorm(first[c], first[i]));
                auto &ldist = distances[i];
                if(dist < ldist) {
                    if(firstround)
                        sumd2 += dist;
                    else
                        sumd2 += dist - ldist;
                    ldist = dist;
                }
            }
        }
        auto sd2i = 1. / sumd2;
        for(IT i = 0; i < np; ++i)
            cdf[i] = distances[i] * sd2i; // Maybe SIMD later?
        std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());
        cdf.back() = 1.;
        auto rv = double(rng()) / rng.max();
        centers.push_back(std::lower_bound(cdf.begin(), cdf.end(), rv) - cdf.begin());
    }
    return centers;
}

} // clustering
