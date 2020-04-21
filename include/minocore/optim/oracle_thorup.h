#pragma once
#include <cmath>
#include <random>
#include <thread>
#include "minocore/util/graph.h"
#include "minocore/util/blaze_adaptor.h"
#include <cassert>
#include "fastiota/fastiota_ho.h"
#include "minocore/util/oracle.h"


namespace minocore {
namespace thorup {

/*
 * Calculates facility centers, costs, and the facility ID to which each point in the dataset is assigned.
 * This could be made iterative by:
 *  1. Performing one iteration.
 *  2. Use the selected points F as the new set of points (``npoints''), with weight = |C_f| (number of cities assigned to facility f)
 *  3. Wrap the previous oracle in another oracle that maps indices within F to the original data
 *  4. Performing the next iteration
 */
template<typename Oracle,
         typename FT=std::decay_t<decltype(std::declval<Oracle>()(0,0))>,
         typename WFT=FT,
         typename IT=uint32_t
        >
std::tuple<std::vector<IT>, blz::DV<FT>, std::vector<IT>>
oracle_thorup_d(const Oracle &oracle, size_t npoints, unsigned k, const WFT *weights=static_cast<const WFT *>(nullptr), double npermult=21, double nroundmult=3, double eps=0.5, uint64_t seed=1337)
{
    const FT total_weight = weights ? static_cast<FT>(blz::sum(blz::CustomVector<WFT, blz::unaligned, blz::unpadded>((WFT *)weights, npoints)))
                                    : static_cast<FT>(npoints);
    size_t nperround = npermult * k * std::log(total_weight) / eps;
#if VERBOSE_AF
    std::fprintf(stderr, "npoints: %zu. total weight: %g. nperround: %zu. Weights? %s\n",
                 npoints, total_weight, nperround, weights ? "true": "false");
#endif

    wy::WyRand<IT, 2> rng(seed);
    blz::DV<FT> mincosts(npoints, std::numeric_limits<FT>::max());   // minimum costs per point
    std::vector<IT> minindices(npoints, IT(-1)); // indices to which points are assigned
    size_t nr = npoints; // Manually managing count
    std::unique_ptr<IT[]> R(new IT[npoints]);
    fastiota::iota(R.get(), npoints, 0);
    std::vector<IT> F;
    shared::flat_hash_set<IT> tmp;
    std::vector<IT> current_batch;
    std::unique_ptr<FT[]> cdf(new FT[nr]);
    std::uniform_real_distribution<WFT> urd;
    auto weighted_select = [&]() {
        return std::lower_bound(cdf.get(), cdf.get() + nr, cdf[nr - 1] * urd(rng)) - cdf.get();
    };
    size_t rounds_to_do = std::ceil(nroundmult * std::log(total_weight));
    //std::fprintf(stderr, "rounds to do: %zu\n", rounds_to_do);
    while(rounds_to_do--) {
        // Sample points not yet added and calculate un-calculated distances
        if(!weights && nr <= nperround) {
            //std::fprintf(stderr, "Adding all\n");
            F.insert(F.end(), R.get(), R.get() + nr);
            for(auto it = R.get(), eit = R.get() + nr; it < eit; ++it) {
                auto v = *it;
                //std::fprintf(stderr, "Adding index %zd/value %u\n", it - R.get(), v);
                mincosts[v] = 0.;
                minindices[v] = v;
                for(size_t j = 0; j < npoints; ++j) {
                    if(j != v && mincosts[j] != 0.) {
                        if(auto score = oracle(v, j);score < mincosts[j]) {
                            mincosts[j] = score;
                            minindices[j] = v;
                        }
                    }
                }
            }
            nr = 0;
        } else {
            // Sample new points, either at random
            if(!weights) {
                //std::fprintf(stderr, "Uniformly sampling to fill tmp\n");
                while(tmp.size() < nperround) {
                    tmp.insert(rng() % nr);
                }
            // or weighted
            } else {
                std::partial_sum(R.get(), R.get() + nr, cdf.get(), [weights](auto csum, auto newv) {
                    return csum + weights[newv];
                });
#if 0
                for(size_t i = 0; i < nr; ++i) {
                    std::fprintf(stderr, "%zu|%g|%g%%\n", i, cdf[i], cdf[i] * 100. / cdf[nr - 1]);
                }
#endif
                if(cdf[nr - 1] <= nperround) {
                    //std::fprintf(stderr, "Adding the rest, nr = %zu, cdf[nr - 1] = %g\n", nr, cdf[nr - 1]);
                    for(IT i = 0; i < nr; ++i) tmp.insert(i);
                } else {
                    WFT weight_so_far = 0;
                    size_t sample_count = 0;
                    while(weight_so_far < nperround && tmp.size() < nr) {
                        ++sample_count;
                        auto ind = weighted_select();
                        if(tmp.find(ind) != tmp.end()) continue;
                        tmp.insert(ind);
                        weight_so_far += weights[R[ind]];
                        //std::fprintf(stderr, "tmp size after growing: %zu. nr: %zu. sample count: %zu. Current weight: %g. Desired weight: %zu\n", tmp.size(), nr, sample_count, weight_so_far, size_t(nperround));
                    }
#if 0
                    std::fprintf(stderr, "Took %zu samples to get %zu items of total weight %g\n", sample_count, tmp.size(), weight_so_far);
#endif
                }
#if 0
                std::fprintf(stderr, "Sampled %zu items of total weight %0.12g\n", tmp.size(),
                             std::accumulate(tmp.begin(), tmp.end(), 0., [&](auto y, auto x) {return y + weights[R[x]];}));
#endif
            }
            // Update F, R, and mincosts/minindices
            current_batch.assign(tmp.begin(), tmp.end());
            tmp.clear();
            for(const auto item: current_batch)
                F.push_back(R[item]);
            shared::sort(current_batch.begin(), current_batch.end(), std::greater<>());
            for(const auto v: current_batch) {
                auto actual_index = R[v];
                minindices[actual_index] = actual_index;
                mincosts[actual_index] = 0.;
                std::swap(R[v], R[--nr]);
                for(size_t j = 0; j < npoints; ++j) {
                    if(j != actual_index) {
                        if(auto oldcost = mincosts[j]; oldcost != 0.) {
                            auto newcost = oracle(actual_index, j);
                            if(newcost < oldcost) {
                                mincosts[j] = newcost;
                                minindices[j] = actual_index;
                            }
                        }
                    }
                }
            }
        }
        // Select pivot and remove others.
        if(nr == 0) break;
        unsigned pivot_index;
        if(weights) {
            std::partial_sum(R.get(), R.get() + nr, cdf.get(), [weights](auto csum, auto newv) {
                return csum + weights[newv];
            });
            pivot_index = weighted_select();
        } else {
            pivot_index = rng() % nr;
        }
        //auto &pivot = R[pivot_index];
        const FT pivot_mincost = mincosts[R[pivot_index]];
        for(auto it = R.get() + nr, e = R.get(); --it >= e;)
            if(auto &v = *it; mincosts[v] <= pivot_mincost)
                std::swap(v, R[--nr]);
    }
#if VERBOSE_AF
    FT final_total_cost = 0.;
    for(size_t i = 0; i < mincosts.size(); ++i) {
        final_total_cost += weights ? mincosts[i] * weights[i]: mincosts[i];
    }
    std::fprintf(stderr, "[LINE %d] Returning solution with mincosts [%zu] and minindices [%zu] with F of size %zu/%zu and final total cost %g for weight %g.\n", __LINE__, mincosts.size(), minindices.size(), F.size(), npoints, final_total_cost, total_weight);
#endif
#if 0
    for(size_t i = 0; i < mincosts.size(); ++i) {
        std::fprintf(stderr, "ID %zu has %g as mincost and %u as minind\n", i, mincosts[i], minindices[i]);
    }
#endif
    return {F, mincosts, minindices};
}

/*
 * Note: iterated_oracle_thorup_d uses the cost *according to the weighted data* from previous iterations,
 * not the cost of the current solution against the original data when selecting which
 * sub-iteration to pursue. This might be change in future iterations.
 */

template<typename Oracle,
         typename FT=std::decay_t<decltype(std::declval<Oracle>()(0,0))>,
         typename WFT=FT,
         typename IT=uint32_t
        >
std::tuple<std::vector<IT>, blz::DV<FT>, std::vector<IT>>
iterated_oracle_thorup_d(const Oracle &oracle, size_t npoints, unsigned k, unsigned num_iter=3, unsigned num_sub_iter=8,
                         const WFT *weights=static_cast<const WFT *>(nullptr), double npermult=21, double nroundmult=3, double eps=0.5, uint64_t seed=1337)
{
    auto getw = [weights](size_t index) {
        return weights ? weights[index]: static_cast<WFT>(1.);
    };
#if !NDEBUG
    const FT total_weight = weights ? blz::sum(blz::CustomVector<WFT, blz::unaligned, blz::unpadded>((WFT *)weights, npoints))
                                    : WFT(npoints);
#endif
    wy::WyHash<uint64_t, 2> rng(seed);
    std::tuple<std::vector<IT>, blz::DV<FT>, std::vector<IT>> ret;
    auto &[centers, costs, bestindices] = ret; // Unpack for named access
    FT best_cost;
    // For convenience: a custom vector
    //                  which is empty if weights is null and full otherwise.
    {
        std::unique_ptr<blz::CustomVector<const WFT, blz::unaligned, blz::unpadded>> wview;
        if(weights) wview.reset(new blz::CustomVector<const WFT, blz::unaligned, blz::unpadded>(weights, npoints));
        auto do_thorup_sample = [&]() {
            return oracle_thorup_d(oracle, npoints, k, weights, npermult, nroundmult, eps, rng());
        };
        auto get_cost = [&](const auto &x) {
            return wview ? blz::dot(x, *wview): blz::sum(x);
        };

        // gather first set of sampled points
        ret = do_thorup_sample();
        best_cost = get_cost(costs);

        // Repeat this process a number of times and select the best-scoring set of points.
        OMP_PFOR
        for(unsigned sub_iter = 0; sub_iter < num_sub_iter; ++sub_iter) {
            auto next_sol = do_thorup_sample();
            auto &[centers2, costs2, bestindices2] = next_sol;
            auto next_cost = get_cost(costs2);
            if(next_cost < best_cost) {
                OMP_CRITICAL
                {
#ifdef _OPENMP
                    // Check again after acquiring the lock in case the value has changed, but only
                    // if parallelized
                    if(next_cost < best_cost)
#endif
                    {
                        ret = std::move(next_sol);
                        best_cost = next_cost;
                    }
                }
            }
        }
    }

    // Calculate weights for center points
    blz::DV<FT> center_weights(centers.size(), FT(0));
    shared::flat_hash_map<IT, IT> asn2id; asn2id.reserve(centers.size());
    for(size_t i = 0; i < centers.size(); asn2id[centers[i]] = i, ++i);
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0; i < npoints; ++i) {
        const auto weight = getw(i);
        auto it = asn2id.find(bestindices[i]);
        assert(it != asn2id.end());
        OMP_ATOMIC
        center_weights[it->second] += weight;
    }
#ifndef NDEBUG
    bool nofails = true;
    for(size_t i = 0; i < center_weights.size(); ++i) {
        if(center_weights[i] <= 0.) {
            std::fprintf(stderr, "weight %zu for center %u is nonpositive: %g and is a center\n", i, centers[i], center_weights[i]);
            nofails = false;
        }
    }
    assert(std::abs(blz::sum(center_weights) - total_weight) < 1e-4 ||
           !std::fprintf(stderr, "Expected sum %g, found %g\n", total_weight, blz::sum(center_weights)));
    assert(nofails);
#endif
    shared::flat_hash_map<IT, IT> sub_asn2id;
    for(size_t iter = 0; iter < num_iter; ++iter) {
        // Setup helpers:
        auto wrapped_oracle = make_oracle_wrapper(oracle, centers); // Remapping old oracle to new points.
        auto do_iter_thorup_sample = [&]() { // Performs wrapped oracle Thorup D
            return oracle_thorup_d(wrapped_oracle, centers.size(), k, center_weights.data(), npermult, nroundmult, eps, rng());
        };
        auto get_cost = [&](const auto &x) { // Calculates the cost of a set of centers.
            return blz::dot(x, center_weights);
            // Can this be easily done using the distance from the full without performing all recalculations?
        };

        // Get first solution
        auto [sub_centers, sub_costs, sub_bestindices] = do_iter_thorup_sample();
        best_cost = get_cost(sub_costs);
        OMP_PFOR
        for(unsigned sub_iter = 0; sub_iter < num_sub_iter; ++sub_iter) {
            auto new_ret = do_iter_thorup_sample();
            if(auto next_cost = get_cost(std::get<1>(new_ret)); next_cost < best_cost) {
                OMP_CRITICAL
                {
#ifdef _OPENMP
                    if(next_cost < best_cost)
#endif
                    {
                        std::tie(sub_centers, sub_costs, sub_bestindices) = std::move(new_ret);
#if VERBOSE_AF
                        std::fprintf(stderr, "[subiter %zu|mainiter %u] iter ret sizes after replacing old cost %g with %g: %zu/%zu/%zu\n",
                                     iter, sub_iter, best_cost, next_cost, sub_centers.size(), sub_costs.size(), sub_bestindices.size());
#endif
                        best_cost = next_cost;
                    }
                }
            }
        }

        // reassign centers and weights
        assert(sub_bestindices.size() == center_weights.size());
        sub_asn2id.clear();
        for(size_t i = 0; i < sub_centers.size(); sub_asn2id[sub_centers[i]] = i, ++i);
        blz::DV<FT> sub_center_weights(sub_centers.size(), FT(0));
        OMP_PFOR
        for(size_t i = 0; i < sub_bestindices.size(); ++i) {
            assert(sub_asn2id.find(sub_bestindices[i]) != sub_asn2id.end());
            const auto weight_idx = sub_asn2id[sub_bestindices[i]];
            auto item_weight = center_weights[i];
            OMP_ATOMIC
            sub_center_weights[weight_idx] += item_weight;
        }

        DBG_ONLY(for(const auto w: sub_center_weights) assert(w > 0.);)
        assert(std::abs(blz::sum(sub_center_weights) - total_weight) <= 1.e-4);

        // Convert back to original coordinates
        auto transform_func = [&wrapped_oracle](auto x) {return wrapped_oracle.lookup(x);};
        std::transform(sub_centers.begin(), sub_centers.end(), sub_centers.begin(),
                       transform_func);
        std::transform(sub_bestindices.begin(), sub_bestindices.end(), sub_bestindices.begin(),
                       transform_func);
        std::tie(centers, center_weights, bestindices)
            = std::tie(sub_centers, sub_center_weights, sub_bestindices);
    }
    return {std::move(centers), std::move(costs), std::move(bestindices)};
}


} // thorup

using thorup::oracle_thorup_d;
using thorup::iterated_oracle_thorup_d;
using thorup::oracle_thorup_d;


} // namespace minocore
