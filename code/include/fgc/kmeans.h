#pragma once
#ifndef FGC_KMEANS_H__
#define FGC_KMEANS_H__
#include <cassert>
#include <mutex>
#include <numeric>
#include "matrix_coreset.h"


namespace coresets {

#ifdef USE_TBB
using std::inclusive_scan;
#endif
using std::partial_sum;
using blz::sqrL2Norm;

template<typename C>
using ContainedTypeFromIterator = std::decay_t<decltype((*std::declval<C>())[0])>;


/*
 * TODO: Adapt using https://arxiv.org/pdf/1309.7109.pdf for arbitrary distances
 */

template<typename Iter, typename FT=ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG, typename Norm=sqrL2Norm>
std::tuple<std::vector<IT>, std::vector<IT>, std::vector<FT>>
kmeanspp(Iter first, Iter end, RNG &rng, size_t k, const Norm &norm=Norm()) {
    static_assert(std::is_floating_point<FT>::value, "FT must be fp");
    size_t np = end - first;
    std::vector<IT> centers;
    std::vector<FT> distances(np, 0.), cdf(np);
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
    std::vector<IT> assignments(np);
    while(centers.size() < k) {
        // At this point, the cdf has been prepared, and we are ready to sample.
        // add new element
        auto newc = std::lower_bound(cdf.begin(), cdf.end(), cdf.back() * double(rng()) / rng.max()) - cdf.begin();
        const auto current_center_id = centers.size();
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
                assignments[i] = current_center_id;
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
    return std::make_tuple(std::move(centers), std::move(assignments), std::move(distances));
}


template<typename FT, bool SO,
         typename IT=std::uint32_t, typename RNG, typename Norm=sqrL2Norm>
auto
kmeanspp(const blaze::DynamicMatrix<FT, SO> &mat, RNG &rng, size_t k, const Norm &norm=Norm(), bool rowwise=true) {
    std::tuple<std::vector<IT>, std::vector<IT>, std::vector<FT>> ret;
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

template<typename IT, typename MatrixType, typename CMatrixType=MatrixType, typename WFT=double>
double lloyd_iteration(std::vector<IT> &assignments, std::vector<WFT> &counts,
                     CMatrixType &centers, MatrixType &data,
                     const WFT *weights=nullptr)
{
    static_assert(std::is_floating_point_v<WFT>, "WTF must be floating point for weighted kmeans");
    // make sure this is only rowwise/rowMajor
    assert(counts.size() == centers.rows());
    assert(centers.columns() == data.columns());
    // 1. Gets means of assignments
    centers = static_cast<typename CMatrixType::ElementType_t>(0.);
    const size_t nc = centers.columns(), nr = data.rows();
    auto getw = [weights](size_t ind) {
        return weights ? weights[ind]: WFT(1.);
    };
    // TODO: parallelize (one thread per center, maybe?)
    for(size_t i = 0; i < nr; ++i) {
        assert(assignments[i] < centers.size());
        auto asn = assignments[i];
        auto dr = row(data, i);
        auto cr = row(centers, asn);
        const auto w = getw(i);
        cr += (dr * w);
        counts[asn] += w;
    }
    for(size_t i = 0; i < centers.rows(); ++i) {
        assert(counts[i]);
        row(centers, i) /= counts[i];
    }
    // 2. Assign centers
    double total_loss = 0.;
    OMP_PRAGMA("omp parallel for reduction(+:total_loss)")
    for(size_t i = 0; i < nr; ++i) {
        auto dr = row(data, i);
        auto dist = std::numeric_limits<double>::max();
        double newdist;
        unsigned label = -1;
        for(size_t j = 0; j < centers.rows(); ++j) {
            if((newdist = sqrL2Dist(dr, row(centers, i))) < dist) {
                dist = newdist;
                label = j;
            }
        }
        assignments[i] = label;
        total_loss += getw(i) * dist;
    }
    return total_loss;
}

template<typename IT, typename MatrixType, typename CMatrixType=MatrixType, typename WFT=double>
void lloyd_loop(std::vector<IT> &assignments, std::vector<WFT> &counts,
                     CMatrixType &centers, MatrixType &data,
                     double tolerance=0., size_t maxiter=-1,
                     const WFT *weights=nullptr)
{
    if(tolerance < 0.) throw 1;
    size_t iternum = 0;
    double oldloss = std::numeric_limits<double>::max(), newloss;
    for(;;) {
        newloss = lloyd_iteration(assignments, counts, centers, data, weights);
        if(iternum++ == maxiter || std::abs(oldloss - newloss) < tolerance)
            break;
        std::fprintf(stderr, "new loss at %zu: %g. old loss: %g\n", iternum, newloss, oldloss);
        oldloss = newloss;
    }
    for(;;) {
        double newloss = lloyd_iteration(assignments, counts, centers, data, weights);
        if(std::abs(oldloss - newloss) / oldloss < tolerance || iternum++ == maxiter) return;
        oldloss = newloss;
    }
}



template<typename Iter,
         typename IT=std::uint32_t, typename RNG=wy::WyRand<uint32_t, 2>,
         typename FT=ContainedTypeFromIterator<Iter>>
auto kmeans_coreset(Iter start, Iter end,
                    size_t k, RNG &rng,
                    size_t cs_size,
                    const FT *weights=nullptr) {
    auto [centers, assignments, sqdists] = kmeanspp(start, end, rng, k, sqrL2Norm());
    
    using sq_t = typename decltype(sqdists)::value_type;
    coresets::CoresetSampler<sq_t, IT> cs;
    size_t np = end - start;
#if 0
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
#endif
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

// TODO: 1. get run kmeans clustering on MatrixCoreset
//       2. Use this for better coreset construction (since coreset size is dependent on the approximation ratio)
//       3. Generate new solution
//       4. Iterate over this
//       5. ???
//       6. PROFIT
//       Why?
//       The solution is 1 + eps accurate, with the error being 1/eps^2
//       We can effectively remove the log(n) approximation 
//       ratio from
//       Epilogue.
//       7. Add mmap accessor


} // namespace coresets
#endif // FGC_KMEANS_H__
