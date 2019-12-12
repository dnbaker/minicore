#pragma once
#include "aesctr/wy.h"
#include <vector>
#include "robin-hood-hashing/src/include/robin_hood.h"
#include "alias_sampler/alias_sampler.h"

                                                                                                    

namespace coresets {
template <typename Key, typename T, typename Hash = robin_hood::hash<Key>,                             
          typename KeyEqual = std::equal_to<Key>, size_t MaxLoadFactor100 = 80>                        
using hash_map = robin_hood::unordered_flat_map<Key, T, Hash, KeyEqual, MaxLoadFactor100>;        
inline namespace sampling {


template<typename Container, typename FT=float, typename IT=std::uint32_t>
auto calculate_sensitivities_and_sample(size_t npoints, const FT *costs, const IT *assignments, size_t ncenters, size_t points_to_sample, const FT *weights=nullptr, uint64_t seed=137) {
    // TODO wrap this in a class?
    std::vector<FT> weight_sums(ncenters);
    std::vector<IT> center_counts(ncenters);
    FT total_cost = 0.;
    auto getweight = [weights](IT ind) {return weights ? weights[ind]: FT(1.);};
    for(size_t i = 0; i < npoints; ++i) {
        auto asn = assignments[i];
        assert(ans < ncenters);
        const auto w = getweight(i);
        weight_sums[asn] += w; // If unweighted, weights are 1.
        total_cost += w * costs[i];
        ++center_counts[asn];
    }
    total_cost *= 2.; // For division
    auto tcinv = 1. / total_cost;
    auto probs = std::make_unique<FT []>(npoints);
    for(auto i = 0u; i < ncenters; ++i)
        weight_sums[i] = 1./(2. * center_counts[i] * weight_sums[i]);
    for(size_t i = 0; i < npoints; ++i) {
        const auto w = getweight(i);
        probs[i] = w * (costs[i] * tcinv + weight_sums[assignments[i]]);
    }
    alias_sampler<FT, IT, wy::WyHash<uint32_t, 2>> sampler(probs.get(), probs.get() + npoints, seed);
    wy::WyRand<uint32_t, 2> rng(seed);
    hash_map<IT, FT> samples;
    samples.reserve(points_to_sample);
    const FT nsamplinv = 1. / points_to_sample;
    do {
        auto ind = sampler();
        auto assigned_weight = getweight(i) * nsamplinv / probs[ind];
        auto it = samples.find(ind);
        if(it == samples.end())
            samples.emplace(ind, assigned_weight);
        else
            it->second += assigned_weight;
    } while(samples.size() < points_to_sample);
    return samples;
}




}

}//coresets


