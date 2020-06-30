#ifndef FGC_OPTIM_KCENTER_H__
#define FGC_OPTIM_KCENTER_H__
#include "minocore/coreset/matrix_coreset.h"
#include "minocore/util/div.h"
#include "minocore/util/blaze_adaptor.h"
#include "minocore/util/fpq.h"

namespace minocore {
namespace coresets {
using std::partial_sum;
using blz::L2Norm;
using blz::sqrL2Norm;
using blz::push_back;
using util::fpq;


/*
 *
 * 2-approximate solution
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
        FT bestcost = std::numeric_limits<FT>::min();
        IT bestind = 0;
        OMP_PFOR
        for(IT i = 0; i < np; ++i) {
            if(!distances[i]) continue;
            auto v = oracle(i, newc);
            distances[i] = std::min(v, distances[i]);
            if(v > bestcost) {
                OMP_ONLY(if(v > bestcost))
                {
                    OMP_CRITICAL
                    {
                        bestcost = v;
                        bestind = i;
                    }
                }
            }
        }
        newc = bestind;
#ifndef NDEBUG
        std::fprintf(stderr, "Current max: %g\n", *std::max_element(distances.begin(), distances.end()));
#endif
        centers.push_back(newc);
        distances[newc] = 0.;
    }
    return centers;
} // kcenter_greedy_2approx

/*
// Algorithm 2 from:
// Greedy Strategy Works for k-Center Clustering with Outliers and Coreset Construction
// Hu Ding, Haikuo Yu, Zixiu Wang
// Z = # outliers
// \gamma = z / n
*/

template<typename Iter, typename FT=shared::ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG, typename Norm=L2Norm>
std::vector<IT>
kcenter_greedy_2approx_outliers(Iter first, Iter end, RNG &rng, size_t k, double eps,
                                double gamma=0.001,
                                const Norm &norm=Norm())
{
    auto dm = make_index_dm(first, norm);
    const size_t np = end - first;
    size_t farthestchunksize = std::ceil((1. + eps) * gamma * np);
    if(farthestchunksize > np) farthestchunksize = np;
    fpq<IT, FT> pq(farthestchunksize);
    auto &pqc = pq.getc();
    //pq.reserve(farthestchunksize + 1);
    std::vector<IT> ret;
    std::vector<FT> distances(np, std::numeric_limits<FT>::max());
    ret.reserve(k);
    auto newc = rng() % np;
    ret.push_back(newc);
    do {
        // Fill pq
#ifdef _OPENMP
    #pragma omp declare reduction (merge : fpq<IT, FT> : omp_out.update(omp_in)) initializer(omp_priv(omp_orig))
    #pragma omp parallel for reduction(merge: pq)
#endif
        for(IT i = 0; i < np; ++i) {
            double dist = distances[i];
            if(dist == 0.) continue;
            double newdist;
            if((newdist = dm(i, newc)) < dist)
                dist = newdist;
            distances[i] = dist;
            pq.add(dist, i);
        }

        // Sample point
        newc = pqc[rng() % farthestchunksize].second;
        ret.push_back(newc);
        pqc.clear();
    } while(ret.size() < k);
    return ret;
}// kcenter_greedy_2approx_outliers

template<typename Oracle, typename FT=std::decay_t<decltype(std::declval<Oracle>()(0,0))>,
         typename IT=std::uint32_t, typename RNG, typename Norm=L2Norm>
std::vector<IT>
kcenter_greedy_2approx_outliers(Oracle &oracle, size_t np, RNG &rng, size_t k, double eps,
                                double gamma=0.001)
{
    size_t farthestchunksize = std::ceil((1. + eps) * gamma * np);
    fpq<IT, FT> pq(farthestchunksize);
    //pq.reserve(farthestchunksize + 1);
    std::vector<IT> ret;
    std::vector<FT> distances(np, std::numeric_limits<FT>::max());
    ret.reserve(k);
    auto newc = rng() % np;
    ret.push_back(newc);
    do {
        //const auto &newel = first[newc];
        // Fill pq
#ifdef _OPENMP
    #pragma omp declare reduction (merge : fpq<IT, FT> : omp_out.update(omp_in)) initializer(omp_priv(omp_orig))
    #pragma omp parallel for reduction(merge: pq)
#endif
        for(IT i = 0; i < np; ++i) {
            double dist = distances[i];
            if(dist == 0.) continue;
            double newdist;
            if((newdist = oracle(i, newc)) < dist)
                dist = newdist;
            distances[i] = dist;
            pq.add(dist, i);
        }

        // Sample point
        newc = pq.getc()[rng() % farthestchunksize].second;
        assert(newc < np);
        ret.push_back(newc);
        pq.getc().clear();
    } while(ret.size() < k);
    return ret;
}// kcenter_greedy_2approx_outliers


template<typename Iter, typename FT=double,
         typename IT=std::uint32_t, typename RNG, typename Norm>
std::vector<IT>
solve_kcenter(Iter first, Iter end, const Norm &norm, RNG &rng, size_t k=50, double eps=1.,
              double gamma=0, int nrep=0)
{
    if(gamma == 0.) {
        auto dm = make_index_dm(first, norm);
        auto ret = kcenter_greedy_2approx(first, end, rng, k, norm);
        auto getcost = [&](const auto &ctrs) {
            using RT = decltype(norm(*first, *first));
            RT maxcost = 0.;
#ifdef _OPENMP
        #pragma omp parallel for reduction(max:maxcost)
#endif
            for(unsigned i = 0; i < end - first; ++i) {
                auto &r = *(first + i);
                auto cit = ctrs.begin();
                RT mmc;
                for(mmc = norm(*cit++, r);cit != ctrs.end();mmc = std::min(norm(*cit++, r)));
                maxcost = std::max(mmc, maxcost);
            }
        };
        auto currentcost = getcost(ret);
        while(nrep-- > 0) {
            auto ret2 = kcenter_greedy_2approx(first, end, rng, k, norm);
            if(auto newcost = getcost(ret2); newcost < ret2)
                 std::tie(ret, currentcost) = std::move(std::tie(ret2, newcost));
        }
    }
    return kcenter_greedy_2approx_outliers<Iter, FT>(first, end, rng, k, eps, gamma, norm);
}

template<typename MT, typename FT=double,
         typename IT=std::uint32_t, typename RNG, typename Norm, bool SO>
std::vector<IT>
solve_kcenter(blaze::Matrix<MT, SO> &matrix, const Norm &norm, RNG &rng, size_t k=50, double eps=1.,
              double gamma=0, int nrep=0)
{
    auto &_mat = ~matrix;
    auto rit = blz::rowiterator(_mat);
    return solve_kcenter<decltype(rit.begin()), FT>(rit.begin(), rit.end(), norm, rng, k, eps, gamma, nrep);
}

} // coresets
using coresets::solve_kcenter;
using coresets::kcenter_greedy_2approx_outliers;
using coresets::kcenter_greedy_2approx;
} // minocore

#endif /* FGC_OPTIM_KCENTER_H__ */
