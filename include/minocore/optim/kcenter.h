#ifndef FGC_OPTIM_KCENTER_H__
#define FGC_OPTIM_KCENTER_H__
#include "minocore/coreset/matrix_coreset.h"
#include "minocore/util/div.h"
#include "minocore/util/blaze_adaptor.h"
#include <queue>

namespace minocore {
namespace coresets {
using std::partial_sum;
using blz::L2Norm;
using blz::sqrL2Norm;
using blz::push_back;


/*
 *
 * Greedy, provable 2-approximate solution
 * T. F. Gonzalez. Clustering to minimize the maximum intercluster distance. Theoretical Computer Science, 38:293-306, 1985.
 */
template<typename Iter, typename FT=shared::ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG, typename Norm=L2Norm>
std::vector<IT>
kcenter_greedy_2approx(Iter first, Iter end, RNG &rng, size_t k, const Norm &norm=Norm(), size_t maxdest=0)
{
    static_assert(sizeof(typename RNG::result_type) >= sizeof(IT), "IT must have the same size as the result type of the RNG");
    static_assert(std::is_arithmetic<FT>::value, "FT must be arithmetic");
    auto dm = make_index_dm(first, norm);
    size_t np = end - first;
    if(maxdest == 0) maxdest = np;
    std::vector<IT> centers(k);
    std::vector<FT> distances(np, 0.);
    FT maxdist = 0;
    IT bestind = 0;
    VERBOSE_ONLY(std::fprintf(stderr, "[%s] Starting kcenter_greedy_2approx\n", __PRETTY_FUNCTION__);)
    auto newc = rng() % maxdest;
    centers[0] = newc;
    distances[newc] = 0.;
#ifdef _OPENMP
    OMP_PFOR
#else
    SK_UNROLL_8
#endif
    for(IT i = 0; i < maxdest; ++i) {
        if(unlikely(i == newc)) continue;
        auto v = dm(newc, i);
        distances[i] = v;
        if(v > maxdist) { OMP_CRITICAL { if(v > maxdist) maxdist = v, bestind = i;} }
    }
    assert(distances[newc] == 0.);
    if(k == 1) return centers;
    centers[1] = newc = bestind;
    distances[newc] = 0.;

    for(size_t ci = 2; ci < std::min(k, np); ++ci) {
        maxdist = -1, bestind = 0;
#ifdef _OPENMP
        OMP_PFOR
#else
        SK_UNROLL_8
#endif
        for(IT i = 0; i < maxdest; ++i) {
            if(unlikely(i == bestind)) continue;
            auto &ldist = distances[i];
            if(!ldist) continue;
            auto dist = dm(newc, i);
            if(dist < ldist)
                ldist = dist;
            if(ldist > maxdist) { OMP_CRITICAL { if(ldist > maxdist) maxdist = ldist, bestind = i;} }
        }
        centers[ci] = newc = bestind;
        distances[newc] = 0.;
    }
    return centers;
} // kcenter_greedy_2approx

template<typename Oracle, typename FT=std::decay_t<decltype(std::declval<Oracle>()(0, 0))>,
         typename IT=std::uint32_t, typename RNG, typename Norm=L2Norm>
std::vector<IT>
kcenter_greedy_2approx(Oracle &oracle, const size_t np, size_t k, RNG &rng)
{
    static_assert(sizeof(typename RNG::result_type) >= sizeof(IT), "IT must have the same size as the result type of the RNG");
    static_assert(std::is_arithmetic<FT>::value, "FT must be arithmetic");
    std::vector<IT> centers;
    std::vector<FT> distances(np, 0.);
    VERBOSE_ONLY(std::fprintf(stderr, "[%s] Starting kcenter_greedy_2approx\n", __PRETTY_FUNCTION__);)
    auto newc = rng() % np;
    centers.push_back(newc);
    if(k == 1) return centers;
    distances[newc] = 0.;
#ifdef _OPENMP
    OMP_PFOR
#else
    SK_UNROLL_8
#endif
    for(IT i = 0; i < np; ++i) {
        if(likely(i != newc)) {
            distances[i] = oracle(i, newc);
        }
    }
    newc = std::max_element(distances.begin(), distances.end()) - distances.begin();
    distances[newc] = 0.;
    centers.push_back(newc);
    if(k == 2) return centers;
    
    while(centers.size() < k) {
        OMP_PFOR
        for(IT i = 0; i < np; ++i) {
            if(!distances[i]) continue;
            auto v = oracle(i, newc);
            distances[i] = std::min(v, distances[i]);
        }
        newc = std::max_element(distances.begin(), distances.end()) - distances.begin();
        std::fprintf(stderr, "Current max: %g\n", *std::max_element(distances.begin(), distances.end()));
        centers.push_back(newc);
        distances[newc] = 0.;
    }
    return centers;
} // kcenter_greedy_2approx

} // coresets
} // minocore

#endif /* FGC_OPTIM_KCENTER_H__ */
