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
    static_assert(sizeof(typename RNG::result_type) == sizeof(IT), "IT must have the same size as the result type of the RNG");
    static_assert(std::is_arithmetic<FT>::value, "FT must be arithmetic");
    auto dm = make_index_dm(first, norm);
    size_t np = end - first;
    if(maxdest == 0) maxdest = np;
    std::vector<IT> centers(k);
    std::vector<FT> distances(np, 0.);
    VERBOSE_ONLY(std::fprintf(stderr, "[%s] Starting kcenter_greedy_2approx\n", __PRETTY_FUNCTION__);)
    {
        auto fc = rng() % maxdest;
        centers[0] = fc;
        distances[fc] = 0.;
        OMP_ELSE(OMP_PFOR, SK_UNROLL_8)
        for(size_t i = 0; i < maxdest; ++i) {
            if(unlikely(i == fc)) continue;
            distances[i] = dm(fc, i);
        }
        assert(distances[fc] == 0.);
    }

    for(size_t ci = 1; ci < k; ++ci) {
        auto it = std::max_element(distances.begin(), distances.end());
        VERBOSE_ONLY(std::fprintf(stderr, "maxelement is %zd from start\n", std::distance(distances.begin(), it));)
        uint64_t newc = it - distances.begin();
        centers[ci] = newc;
        distances[newc] = 0.;
        OMP_PFOR
        for(IT i = 0; i < maxdest; ++i) {
            if(unlikely(i == newc)) continue;
            auto &ldist = distances[i];
            const auto dist = dm(newc, i);
            if(dist < ldist) {
                ldist = dist;
            }
        }
        assert(std::find_if(distances.begin(), distances.end(), [](auto x) {return std::isnan(x) || std::isinf(x);})
               == distances.end());
    }
    return centers;
} // kcenter_greedy_2approx

} // coresets
} // minocore

#endif /* FGC_OPTIM_KCENTER_H__ */
