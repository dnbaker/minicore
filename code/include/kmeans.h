#pragma once
#include <cassert>
#include <map>
#include <mutex>
#include <numeric>
#include <vector>
#include "aesctr/wy.h"
#include "matrix_coreset.h"

#if defined(USE_TBB)
#include <execution>
#  define inclusive_scan(x, y, z) inclusive_scan(::std::execution::par_unseq, x, y, z)
#else
#  define inclusive_scan(x, y, z) ::std::partial_sum(x, y, z)
#endif

namespace clustering {

using std::inclusive_scan;
using std::partial_sum;
using blz::sqrL2Norm;

template<typename C>
using ContainedTypeFromIterator = std::decay_t<decltype((*std::declval<C>())[0])>;


/*
 * TODO: Adapt using https://arxiv.org/pdf/1309.7109.pdf for arbitrary distances
 */

template<typename Iter, typename FT=ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG, typename Norm=sqrL2Norm>
std::pair<std::vector<IT>, std::vector<FT>>
kmeanspp(Iter first, Iter end, RNG &rng, size_t k, const Norm &norm=Norm()) {
    static_assert(std::is_floating_point<FT>::value, "FT must be fp");
    size_t np = end - first;
    std::vector<IT> centers;
    std::vector<FT> distances(np, 0.), cdf(np);
    //std::vector<IT> assignments(np, IT(-1));
    double sumd2 = 0.;
    {
        auto fc = rng() % np;
        centers.push_back(fc);
        const auto &lhs = first[centers.front()];
#ifdef _OPENMP
        OMP_PRAGMA("omp parallel for reduction(+:sumd2)")
        for(size_t i = 0; i < np; ++i) {
            double dist = norm(lhs, first[i]);
            distances[i] = dist;
            sumd2 += dist;
        }
#else
        SK_UNROLL_8
        for(size_t i = 0; i < np; ++i) {
            double dist = norm(lhs, first[i]);
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
        
    while(centers.size() < k) {
        // At this point, the cdf has been prepared, and we are ready to sample.
        // add new element
        auto newc = std::lower_bound(cdf.begin(), cdf.end(), cdf.back() * double(rng()) / rng.max()) - cdf.begin();
        centers.push_back(newc);
        const auto &lhs = first[newc];
        sumd2 -= distances[newc];
        distances[newc] = 0.;
        double sum = sumd2;
        OMP_PRAGMA("omp parallel for reduction(+:sum)")
        for(IT i = 0; i < np; ++i) {
            auto &ldist = distances[i];
            double dist = norm(lhs, first[i]);
            if(dist < ldist) { // Only write if it changed
                auto diff = dist - ldist;
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
    return std::make_pair(std::move(centers), std::move(distances));
}
template<typename FT, bool SO,
         typename IT=std::uint32_t, typename RNG, typename Norm=sqrL2Norm>
auto
kmeanspp(const blaze::DynamicMatrix<FT, SO> &mat, RNG &rng, size_t k, const Norm &norm=Norm(), bool rowwise=true) {
    std::pair<std::vector<IT>, std::vector<FT>> ret;
    const auto &blzview = reinterpret_cast<const blz::DynamicMatrix<FT, SO> &>(mat);
    if(rowwise) {
        auto rowit = blzview.rowiterator();
        ret = kmeanspp(rowit.begin(), rowit.end(), rng, k, norm);
    } else { // columnwise
        auto columnit = blzview.columniterator();
        ret = kmeanspp(columnit.begin(), columnit.end(), rng, k, norm);
    }
    return ret;
}


template<typename Iter,
         typename IT=std::uint32_t, typename RNG=wy::WyRand<uint32_t, 2>,
         typename FT=ContainedTypeFromIterator<Iter>>
auto kmeans_coreset(Iter start, Iter end,
                    size_t k, RNG &rng,
                    size_t cs_size,
                    const FT *weights=nullptr) {
    auto [centers, sqdists] = kmeanspp(start, end, rng, k, sqrL2Norm());
    using sq_t = typename decltype(sqdists)::value_type;
    coresets::CoresetSampler<sq_t, IT> cs;
    size_t np = end - start;
    std::vector<IT> assignments(np);
    // Get assignments
    OMP_PRAGMA("parallel for")
    for(size_t i = 0; i < np; ++i) {
        double minv = std::numeric_limits<double>::max();
        unsigned assign_index;
        for(unsigned j = 0; j < centers.size(); ++j) {
            double newdist;
            if((newdist = blz::l2Dist(start[i], start[centers[j]])) < minv) {
                minv = newdist;
                assign_index = j;
            }
        }
        sqdists[i] = minv;
        assignments[i] = assign_index;
    }
    cs.make_sampler(np, centers.size(), sqdists.data(), assignments.data(), weights,
                    /*seed=*/rng());
    coresets::IndexCoreset<IT, sq_t> ics(cs.sample(cs_size, rng()));
    return ics;
}
template<typename FT, bool SO,
         typename IT=std::uint32_t, typename RNG=wy::WyRand<uint32_t, 2>>
auto kmeans_matrix_coreset(const blaze::DynamicMatrix<FT, SO> &mat, size_t k, RNG &rng, size_t cs_size,
                           const FT *weights=nullptr, bool rowwise=true)
{
    if(!rowwise) throw std::runtime_error("Not implemented");
    const auto &blzview = reinterpret_cast<const blz::DynamicMatrix<FT, SO> &>(mat);
    auto ics = kmeans_coreset(blzview.rowiterator().begin(), blzview.rowiterator().end(),
                              k, rng, cs_size, weights);
    coresets::MatrixCoreset<blaze::DynamicMatrix<FT, SO>, FT> csmat = index2matrix(ics, mat);
    return csmat;
}
#undef inclusive_scan

} // clustering
