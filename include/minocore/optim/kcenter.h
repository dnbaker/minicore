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
    static constexpr FT startval =  std::is_floating_point<FT>::value ? -std::numeric_limits<FT>::max(): std::numeric_limits<FT>::min();
    std::pair<FT, IT> maxdist(startval, 0);
    VERBOSE_ONLY(std::fprintf(stderr, "[%s] Starting kcenter_greedy_2approx\n", __PRETTY_FUNCTION__);)
    auto newc = rng() % maxdest;
    centers[0] = newc;
    distances[newc] = 0.;
#ifdef _OPENMP
    #pragma omp declare reduction (max : std::pair<FT, IT> : std::max(omp_in, omp_out) )
    #pragma omp parallel for reduction(max: maxdist)
#else
    SK_UNROLL_8
#endif
    for(IT i = 0; i < maxdest; ++i) {
        if(likely(i != newc)) {
            auto v = dm(newc, i);
            distances[i] = v;
            maxdist = std::max(maxdist, std::make_pair(v, i));
        }
    }
    assert(distances[newc] == 0.);
    if(k == 1) return centers;
    centers[1] = newc = maxdist.second;
    distances[newc] = 0.;

    for(size_t ci = 2; ci < k; ++ci) {
        maxdist = std::pair<FT, IT>(startval, 0);
#ifdef _OPENMP
        #pragma omp declare reduction (max : std::pair<FT, IT> : std::max(omp_in, omp_out) )
        #pragma omp parallel for reduction(max: maxdist)
#else
        SK_UNROLL_8
#endif
        for(IT i = 0; i < maxdest; ++i) {
            if(unlikely(i == newc)) continue;
            auto &ldist = distances[i];
            if(!ldist) continue;
            const auto dist = dm(newc, i);
            if(dist < ldist)
                ldist = dist;
            maxdist = std::max(maxdist, std::make_pair(ldist, i));
        }
        centers[ci] = newc = maxdist.second;
        distances[newc] = 0.;
    }
    return centers;
} // kcenter_greedy_2approx

template<typename Oracle, typename FT=std::decay_t<decltype(std::declval<Oracle>()(0, 0))>,
         typename IT=std::uint32_t, typename RNG, typename Norm=L2Norm>
std::vector<IT>
kcenter_greedy_2approx(Oracle &oracle, const size_t np, size_t k, RNG &rng, size_t maxdest=0)
{
    static_assert(sizeof(typename RNG::result_type) >= sizeof(IT), "IT must have the same size as the result type of the RNG");
    static_assert(std::is_arithmetic<FT>::value, "FT must be arithmetic");
    std::vector<IT> centers(k);
    std::vector<FT> distances(np, 0.);
    static constexpr FT startval =  std::is_floating_point<FT>::value ? -std::numeric_limits<FT>::max(): std::numeric_limits<FT>::min();
    std::pair<FT, IT> maxdist(startval, 0);
    VERBOSE_ONLY(std::fprintf(stderr, "[%s] Starting kcenter_greedy_2approx\n", __PRETTY_FUNCTION__);)
    auto newc = rng() % np;
    centers[0] = newc;
    distances[newc] = 0.;
#ifdef _OPENMP
    #pragma omp declare reduction (max : std::pair<FT, IT> : std::max(omp_in, omp_out) )
    #pragma omp parallel for reduction(max: maxdist)
#else
    SK_UNROLL_8
#endif
    for(IT i = 0; i < np; ++i) {
        if(likely(i != newc)) {
            auto v = oracle(i, newc);
            distances[i] = v;
            maxdist = std::max(maxdist, std::make_pair(v, i));
        }
    }
    assert(distances[newc] == 0.);
    if(k == 1) return centers;
    centers[1] = newc = maxdist.second;
    distances[newc] = 0.;

    for(size_t ci = 2; ci < k; ++ci) {
        maxdist = std::pair<FT, IT>(startval, 0);
#ifdef _OPENMP
        #pragma omp declare reduction (max : std::pair<FT, IT> : std::max(omp_in, omp_out) )
        #pragma omp parallel for reduction(max: maxdist)
#else
        SK_UNROLL_8
#endif
        for(IT i = 0; i < np; ++i) {
            if(unlikely(i == newc)) continue;
            auto &ldist = distances[i];
            if(!ldist) continue;
            const auto dist = oracle(i, newc);
            if(dist < ldist) ldist = dist;
            maxdist = std::max(maxdist, std::make_pair(ldist, i));
        }
        centers[ci] = newc = maxdist.second;
        distances[newc] = 0.;
    }
    return centers;
} // kcenter_greedy_2approx

} // coresets
} // minocore

#endif /* FGC_OPTIM_KCENTER_H__ */
