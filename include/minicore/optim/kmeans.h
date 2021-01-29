#pragma once
#ifndef FGC_KMEANS_H__
#define FGC_KMEANS_H__
#include <cassert>
#include <iostream>
#include <mutex>
#include <numeric>
#include "minicore/coreset/matrix_coreset.h"
#include "minicore/util/oracle.h"
#include "minicore/util/timer.h"
#include "minicore/util/div.h"
#include "minicore/optim/lsearchpp.h"
#include "minicore/util/blaze_adaptor.h"
#include "minicore/util/tsg.h"
#include "libsimdsampling/simdsampling.ho.h"
#include "reservoir/include/DOGS/reservoir.h"
#if USE_TBB
#endif

namespace minicore {




namespace coresets {


using std::partial_sum;
using blz::sqrL2Norm;


/*
 * However, using https://arxiv.org/abs/1508.05243 (Strong Coresets for Hard and Soft Bregman Clustering with Applications to Exponential Family Mixtures)
 * any squared Bregman divergence will work for the kmeanspp, including regular exponential families.
 * See http://www.jmlr.org/papers/volume6/banerjee05b/banerjee05b.pdf
 * http://www.cs.utexas.edu/users/inderjit/Talks/bregtut.pdf
 * and perhaps https://arxiv.org/pdf/1309.7109.pdf.
 * The Banerjee paper has a table of relevant information.
 */

/*
 *
 * oracle: computes distance D(x, y) for i in [0, np)
 * weights: null if equal, used if provided
 * lspprounds: how many localsearch++ rounds to perform. By default, perform none.
 */


template<typename Oracle, typename FT=double,
         typename IT=std::uint32_t, typename RNG, typename WFT=FT>
auto
kmeanspp(const Oracle &oracle, RNG &rng, size_t np, size_t k, const WFT *weights=nullptr, size_t lspprounds=0, bool use_exponential_skips=false, bool parallelize_oracle=true, size_t n_local_samples=1)
{
    const bool emit_log = false;
    if(emit_log)
        std::fprintf(stderr, "Starting kmeanspp with np = %zu and k = %zu%s and %s, and %s.\n", np, k, weights ? " and non-null weights": "", parallelize_oracle ? "parallelized": "unparallelized", use_exponential_skips ? "with exponential skips": "SIMD sampling");
    std::vector<IT> centers(k, IT(0));
    blz::DV<FT> distances(np, std::numeric_limits<FT>::max()), odists;
    {
        IT fc;
        auto setdists = [&](auto &vec, auto index) {
            if(parallelize_oracle) {
                vec = blaze::generate(np,[&](auto i) __attribute__((always_inline)) {
                    if(unlikely(i == index)) return FT(0.);
                    return FT(oracle(index, i));
                });
            } else {
                for(size_t i = 0; i < np; ++i) {
                    if(i == index) vec[i] = 0.;
                    else       vec[i] = oracle(index, i);
                }
            }
        };
        fc = rng() % np;
        setdists(distances, fc);
        if(n_local_samples > 1) {
            double fcost = sum(distances);
            if(odists.size() != distances.size()) odists.resize(distances.size());
            for(size_t i = 1; i < n_local_samples; ++i) {
                const size_t ofc = rng() % np;
                setdists(odists, ofc);
                if(double ocost = sum(odists);ocost < fcost) {
                    std::swap(distances, odists);
                    fcost = ocost; fc = ofc;
                }
            }
        }
        centers[0] = fc;
        assert(distances[fc] == 0.);
    }
    std::vector<IT> assignments(np, IT(0));
    std::uniform_real_distribution<double> urd;
    blz::DV<FT> rvals;
    // Short of re-writing the loop fully with SIMD-optimized argmin
    // and performing one single pass through all the data
    // (which is less important if dimensionaliy is high)
    // this is as optimized as it can be.
    // At least it's all embarassingly parallelizable
    const SampleFmt fmt = use_exponential_skips ? USE_EXPONENTIAL_SKIPS: NEITHER;
    blz::SmallArray<uint64_t, 6> samplesbuf(n_local_samples <= 1u ? size_t(0): n_local_samples);
    blz::DV<double> samplescosts(odists.size());
    std::vector<IT> samplesasn(odists.size()), oasn(odists.size());
    if(odists.size()) odists = distances, oasn = assignments;
    for(size_t center_idx = 1;center_idx < k;) {
        //std::fprintf(stderr, "Centers size: %zu/%zu. Newest center: %u\r\n", center_idx, size_t(k), centers[center_idx - 1]);
        // At this point, the cdf has been prepared, and we are ready to sample.
        // add new element
        auto cd = centers.data(), ce = cd + center_idx;
        IT newc = std::numeric_limits<IT>::max();
        auto rngv = rng();
        if(weights) {
            auto w = blz::make_cv(weights, np);
            if constexpr(blaze::TransposeFlag_v<decltype(w)> == blaze::TransposeFlag_v<blz::DV<FT>>)
                rvals = w * distances;
            else
                rvals = trans(w) * distances;
            if(n_local_samples <= 1u)
                newc = reservoir_simd::sample(rvals.data(), np, rngv, fmt);
            else
                reservoir_simd::sample_k(rvals.data(), np, n_local_samples, samplesbuf.data(), rngv, fmt);
        } else {
            if(n_local_samples <= 1u)
                newc = reservoir_simd::sample(distances.data(), np, rngv, fmt);
            else
                reservoir_simd::sample_k(distances.data(), np, n_local_samples, samplesbuf.data(), rngv, fmt);
        }
        double dsum = -1.;
        if(n_local_samples > 1) {
            for(size_t i = 0; i < n_local_samples; ++i) {
                VERBOSE_ONLY(std::fprintf(stderr, "Performing %zu sample/%zu for %zu\n", i, n_local_samples, center_idx);)
                if(i > 0) samplescosts = odists, samplesasn = oasn;
                auto sptr = i > 0 ? &samplescosts: &distances;
                auto iptr = i > 0 ? &samplesasn: &assignments;
                auto nextc = samplesbuf[i];
                if(distances[nextc] <= 0.) std::fprintf(stderr, "Note: Selected item with weight 0 at index %zu\n", i);
                double nsum = 0.;
                if(parallelize_oracle) {
                    OMP_PRAGMA("omp parallel for simd reduction(+:nsum)")
                    for(size_t j = 0; j < np; ++j) {
                        auto &v = sptr->operator[](j);
                        if(auto newv = oracle(nextc, j);newv < v)
                            v = newv, (*iptr)[j] = center_idx;
                        nsum += v;
                    }
                } else {
                    for(size_t j = 0; j < np; ++j) {
                        auto &v = sptr->operator[](j);
                        if(auto newv = oracle(nextc, j);newv < v)
                            v = newv, (*iptr)[j] = center_idx;
                    }
                    nsum = sum(*sptr);
                }
                if(dsum < 0.) {
                    dsum = nsum, newc = nextc;
                } else {
                    if(nsum < dsum) {
                        std::fprintf(stderr, "Sample %zu for idx %zu replaced cost %.10g with %0.10g\n", i, center_idx, nsum, dsum);
                        distances = samplescosts, dsum = nsum, assignments = samplesasn, newc = nextc;
                    } VERBOSE_ONLY(else std::fprintf(stderr, "new cost: %g. old: %g. old cost is %g better\n", nsum, dsum, dsum - nsum);)
                }
            }
            centers[center_idx] = newc;
            assignments[newc] = center_idx;
        } else {
            if(distances[newc] == 0.) std::fprintf(stderr, "Warning: distance of 0 selected. Are all distances 0?\n");
            if(std::find(cd, ce, newc) != ce) {
                std::fprintf(stderr, "Re-selected existing center %u. Continuing...\n", int(newc));
                continue;
            }
            assignments[newc] = center_idx;
            centers[center_idx] = newc;
#define COMPUTE_X(i) do {\
        if(i != newc) {\
            auto &ldist = distances[i];\
            if(ldist <= 0.) continue;\
            if(auto dist = oracle(newc, i); dist < ldist) assignments[i] = center_idx, ldist = dist;\
        }\
    } while(0)
            if(parallelize_oracle) {
                OMP_PFOR_DYN
                for(size_t i = 0; i < np; ++i) {COMPUTE_X(i);}
            } else {
                for(size_t i = 0; i < np; ++i) {COMPUTE_X(i);}
            }
#undef COMPUTE_LOOP
        }
        distances[newc] = 0.;
        ++center_idx;
    }

    if(emit_log) std::fprintf(stderr, "Completed kmeans++ with centers of size %zu\n", centers.size());
    if(lspprounds > 0) {
        if(emit_log) std::fprintf(stderr, "Performing %u rounds of ls++\n", int(lspprounds));
        localsearchpp_rounds(oracle, rng, distances, centers, assignments, np, lspprounds, weights, parallelize_oracle);
    }
    //if(emit_log) std::fprintf(stderr, "returning %zu centers and %zu assignments\n", centers.size(), assignments.size());
    return std::make_tuple(std::move(centers), std::move(assignments), std::vector<FT>(distances.begin(), distances.end()));
}

template<typename Iter, typename FT=double,
         typename IT=std::uint32_t, typename RNG, typename Norm=sqrL2Norm, typename WFT=FT>
auto
kmeanspp(Iter first, Iter end, RNG &rng, size_t k, const Norm &norm=Norm(), WFT *weights=nullptr, size_t lspprounds=0, bool use_exponential_skips=false, bool parallelize_oracle=true, size_t n_local_trials=1) {
    auto dm = make_index_dm(first, norm);
    static_assert(std::is_floating_point<FT>::value, "FT must be fp");
    return kmeanspp<decltype(dm), FT>(dm, rng, end - first, k, weights, lspprounds, use_exponential_skips, parallelize_oracle, n_local_trials);
}

template<typename Oracle, typename FT=double,
         typename IT=std::uint32_t, typename RNG, typename WFT=FT>
std::tuple<std::vector<IT>, std::vector<IT>, std::vector<FT>>
reservoir_kmeanspp(const Oracle &oracle, RNG &rng, size_t np, size_t k, WFT *weights=static_cast<WFT *>(nullptr), int lspprounds=0, int ntimes=1);

template<typename Iter, typename FT=double,
         typename IT=std::uint32_t, typename RNG, typename Norm=sqrL2Norm, typename WFT=FT>
std::tuple<std::vector<IT>, std::vector<IT>, std::vector<FT>>
reservoir_kmeanspp(Iter first, Iter end, RNG &rng, size_t k, const Norm &norm=Norm(), WFT *weights=nullptr, size_t lspprounds=0, int ntimes=1) {
    auto dm = make_index_dm(first, norm);
    static_assert(std::is_floating_point<FT>::value, "FT must be fp");
    return reservoir_kmeanspp<decltype(dm), FT>(dm, rng, end - first, k, weights, lspprounds, ntimes);
}


template<typename Oracle, typename Sol, typename FT=double, typename IT=uint32_t>
std::pair<blaze::DynamicVector<IT>, blaze::DynamicVector<FT>> get_oracle_costs(const Oracle &oracle, size_t np, const Sol &sol)
{
    blaze::DynamicVector<IT> assignments(np);
    blaze::DynamicVector<FT> costs(np, std::numeric_limits<FT>::max());
    OMP_PFOR_DYN
    for(size_t i = 0; i < np; ++i) {
        auto it = sol.begin(), e = sol.end();
        auto mincost = oracle(*it, i);
        IT minind = 0, cind = 0;
        while(++it != e) {
            if(auto newcost = oracle(*it, i); newcost < mincost)
                mincost = newcost, minind = cind;
            ++cind;
        }
        costs[i] = mincost;
        assignments[i] = minind;
    }
    //std::fprintf(stderr, "Centers have total cost %g\n", blz::sum(costs));
    return std::make_pair(assignments, costs);
}


template<typename Oracle, typename FT,
         typename IT, typename RNG, typename WFT>
std::tuple<std::vector<IT>, std::vector<IT>, std::vector<FT>>
reservoir_kmeanspp(const Oracle &oracle, RNG &rng, size_t np, size_t k, WFT *weights, int lspprounds, int ntimes)
{
    schism::Schismatic<IT> div(np);
    std::vector<IT> centers({div.mod(IT(rng()))});
    std::vector<IT> asn(np, 0);
    std::vector<FT> distances(np);
    for(unsigned i = 0; i < np; ++i) {
        distances[i] = oracle(centers[0], i);
    }
    // Helper function for minimum distance
    auto mindist = [&centers,&oracle,&asn,&distances](auto newind) {
        auto cd = centers.data(), it = cd, end = &*centers.end();
        auto dist = oracle(*it, newind);
        auto bestind = 0u;
        while(++it != end) {
            auto newdist = oracle(*it, newind);
            if(newdist < dist) {
                dist = newdist, bestind = it - cd;
            }
        }
        asn[newind] = bestind;
        distances[newind] = dist;
        return dist;
    };
    shared::flat_hash_set<IT> hashset(centers.begin(), centers.end());

    while(centers.size() < k) {
        size_t x;
        do x = rng() % np; while(hashset.find(x) != hashset.end());
        double xdist = mindist(x);
        auto xdi = 1. / xdist;
        const auto baseseed = IT(rng());
        const double max64inv = 1. / std::numeric_limits<uint64_t>::max();
        auto lfunc = [&](unsigned j) {
            if(hashset.find(j) == hashset.end() || !distances[j]) return;
            uint64_t local_seed = baseseed + j;
            wy::wyhash64_stateless(&local_seed);
            auto ydist = mindist(j);
            if(weights) {
                ydist *= weights[j];
            }
            const auto urd_val = local_seed * max64inv;
            if(ydist * xdi > urd_val) {
                OMP_CRITICAL
                {
                    if(ydist * xdi > urd_val)
                        x = j, xdist = ydist, xdi = 1. / xdist;
                    DBG_ONLY(std::fprintf(stderr, "Now x is %d with cost %g\n", int(x), xdist);)
                }
            }
        };
        for(int i = ntimes; i--;) {
            OMP_PFOR
            for(unsigned j = 1; j < np; ++j) {
                lfunc(div.mod(j + x));
            }
        }
        centers.emplace_back(x);
        hashset.insert(x);
    }
    if(lspprounds > 0) {
        localsearchpp_rounds(oracle, rng, distances, centers, asn, np, lspprounds, weights);
    }
    return std::make_tuple(std::move(centers), std::move(asn), std::move(distances));
}

/*
 * Implementation of the $KMC^2$ algorithm from:
 * Bachem, et al. Approximate K-Means++ in Sublinear Time (2016)
 * Available at https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12147/11759
 */


template<typename Oracle,
         typename IT=std::uint32_t, typename RNG>
std::vector<IT>
kmc2(const Oracle &oracle, RNG &rng, size_t np, size_t k, size_t m = 2000)
{
    if(m == 0)  {
        std::fprintf(stderr, "m must be nonzero\n");
        std::abort();
    }
    schism::Schismatic<IT> div(np);
    shared::flat_hash_set<IT> centers{div.mod(IT(rng()))};
    if(*centers.begin() > np) {
        std::fprintf(stderr, "Out of range\n");
        std::abort();
    }
    // Helper function for minimum distance
    auto mindist = [&centers,&oracle](auto newind) {
        typename shared::flat_hash_set<IT>::const_iterator it = centers.begin(), end = centers.end();
        auto dist = oracle(*it, newind);
        while(++it != end) {
            dist = std::min(dist, oracle(*it, newind));
        }
        return dist;
    };

    while(centers.size() < k) {
        auto x = div.mod(IT(rng()));
        double xdist = mindist(x);
        auto xdi = 1. / xdist;
        auto baseseed = IT(rng());
        const double max64inv = 1. / std::numeric_limits<uint64_t>::max();
        auto lfunc = [&](unsigned j) ALWAYS_INLINE {
            if(centers.find(j) != centers.end()) return;
            uint64_t local_seed = baseseed + j;
            wy::wyhash64_stateless(&local_seed);
            auto y = div.mod(local_seed);
            assert(uint32_t(local_seed) % np == y || !std::fprintf(stderr, "seed: %zu. np: %zu. mod: %u. found: %u\n", size_t(local_seed), np, unsigned(local_seed % np), y));
            auto ydist = mindist(y);
            wy::wyhash64_stateless(&local_seed);
            const auto urd_val = local_seed * max64inv;
            if(ydist * xdi > urd_val) {
                OMP_CRITICAL
                {
                    if(ydist * xdi > urd_val)
                        x = y, xdist = ydist, xdi = 1. / xdist;
                }
            }
        };
        OMP_PFOR
        for(unsigned j = 1; j < m; ++j) {
            lfunc(j);
        }
        centers.insert(x);
    }
    return std::vector<IT>(centers.begin(), centers.end());
}


template<typename Iter, typename FT=shared::ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG, typename Norm=sqrL2Norm>
std::vector<IT>
kmc2(Iter first, Iter end, RNG &rng, size_t k, size_t m = 2000, const Norm &norm=Norm()) {
    if(m == 0) throw std::invalid_argument("m must be nonzero");
    auto dm = make_index_dm(first, norm);
    static_assert(std::is_floating_point<FT>::value, "FT must be fp");
    size_t np = end - first;
    return kmc2(dm, rng, np, k, m);
}


template<typename MT, bool SO,
         typename IT=std::uint32_t, typename RNG, typename Norm=sqrL2Norm, typename WFT=typename MT::ElementType>
auto
kmeanspp(const blaze::Matrix<MT, SO> &mat, RNG &rng, size_t k, const Norm &norm=Norm(), bool rowwise=true, const WFT *weights=nullptr, size_t lspprounds=0, bool use_exponential_skips=false, bool parallelize_oracle=blaze::IsDenseMatrix_v<MT>, size_t n_local_samples=1) {
    if(rowwise) {
        auto rowit = blz::rowiterator(*mat);
        return kmeanspp(rowit.begin(), rowit.end(), rng, k, norm, weights, lspprounds, use_exponential_skips, parallelize_oracle, n_local_samples);
    } else { // columnwise
        auto columnit = blz::columniterator(*mat);
        return kmeanspp(columnit.begin(), columnit.end(), rng, k, norm, weights, lspprounds, use_exponential_skips, parallelize_oracle, n_local_samples);
    }
}

template<typename MT, bool SO,
         typename IT=std::uint32_t, typename RNG, typename Norm=sqrL2Norm, typename WFT=typename MT::ElementType>
auto
reservoir_kmeanspp(const blaze::Matrix<MT, SO> &mat, RNG &rng, size_t k, const Norm &norm=Norm(), bool rowwise=true, const WFT *weights=nullptr, size_t lspprounds=0, int ntimes=1) {
    using FT = typename MT::ElementType;
    std::tuple<std::vector<IT>, std::vector<IT>, std::vector<FT>> ret;
    if(rowwise) {
        auto rowit = blz::rowiterator(*mat);
        ret = reservoir_kmeanspp(rowit.begin(), rowit.end(), rng, k, norm, weights, lspprounds, ntimes);
    } else { // columnwise
        auto columnit = blz::columniterator(*mat);
        ret = reservoir_kmeanspp(columnit.begin(), columnit.end(), rng, k, norm, weights, lspprounds, ntimes);
    }
    return ret;
}

template<typename MT, bool SO,
         typename IT=std::uint32_t, typename RNG, typename Norm=sqrL2Norm>
auto
kmc2(const blaze::Matrix<MT, SO> &mat, RNG &rng, size_t k,
     size_t m=2000,
     const Norm &norm=Norm(),
     bool rowwise=true)
{
    std::vector<IT> ret;
    if(rowwise) {
        auto rowit = blz::rowiterator(*mat);
        ret = kmc2(rowit.begin(), rowit.end(), rng, k, m, norm);
    } else { // columnwise
        auto columnit = blz::columniterator(*mat);
        ret = kmc2(columnit.begin(), columnit.end(), rng, k, m, norm);
    }
    return ret;
}

template<typename IT, typename MatrixType, typename CMatrixType=MatrixType, typename CFT, typename WFT=double, typename Functor=blz::sqrL2Norm>
double lloyd_iteration(std::vector<IT> &assignments, std::vector<CFT> &counts,
                       CMatrixType &centers, MatrixType &data,
                       const Functor &func=Functor(),
                       const WFT *weights=static_cast<WFT *>(nullptr),
                       uint64_t seed=0,
                       bool use_moving_average=false)
{
    static_assert(std::is_floating_point_v<WFT>, "WTF must be floating point for weighted kmeans");
    std::mt19937_64 mt(seed);
    // make sure this is only rowwise/rowMajor
    assert(counts.size() == centers.rows() || !std::fprintf(stderr, "counts size: %zu. centers rows: %zu\n", counts.size(), centers.rows()));
    assert(centers.columns() == data.columns());
    // 1. Gets means of assignments
    const size_t nr = data.rows();
    auto getw = [weights](size_t ind) {
        return weights ? weights[ind]: WFT(1.);
    };
    OMP_ONLY(std::unique_ptr<std::mutex[]> mutexes = std::make_unique<std::mutex[]>(centers.rows());)
    centers = static_cast<typename CMatrixType::ElementType>(0.);
    std::fill(counts.data(), counts.data() + counts.size(), WFT(0.));
    assert(blz::sum(centers) == 0.);
    bool centers_reassigned;
    std::unique_ptr<typename MatrixType::ElementType[]> costs;
    get_assignment_counts:
    centers_reassigned = false;
    /*
     *
     * The moving average is supposed to be more numerically stable, but I get better results
     * with naive summation.
     */
    if(!use_moving_average) {
        OMP_PRAGMA("omp parallel for schedule(dynamic)")
        for(size_t i = 0; i < nr; ++i) {
            assert(assignments[i] < centers.rows());
            auto asn = assignments[i];
            auto dr = row(data, i BLAZE_CHECK_DEBUG);
            auto cr = row(centers, asn BLAZE_CHECK_DEBUG);
            const auto w = getw(i);
            {
                OMP_ONLY(std::lock_guard<std::mutex> lg(mutexes[asn]);)
                if(w == 1.) {
                    blz::serial(cr.operator+=(dr));
                }
                else blz::serial(cr.operator+=(dr * w));
            }
            OMP_ATOMIC
            counts[asn] += w;
        }
        OMP_PFOR
        for(size_t i = 0; i < centers.rows(); ++i)
            row(centers, i BLAZE_CHECK_DEBUG) *= (1. / counts[i]);
    } else {
        OMP_PRAGMA("omp parallel for schedule(dynamic)")
        for(size_t i = 0; i < nr; ++i) {
            assert(assignments[i] < centers.rows());
            auto asn = assignments[i];
            auto dr = row(data, i BLAZE_CHECK_DEBUG);
            auto cr = row(centers, asn BLAZE_CHECK_DEBUG);
            const auto w = getw(i);
            {
                OMP_ONLY(std::lock_guard<std::mutex> lg(mutexes[asn]);)
                auto oldw = counts[asn];
                if(!oldw) {
                    cr = dr;
                } else {
                    cr += (dr - cr) * (w / (oldw + w));
                }
                counts[asn] = oldw + w;
            }
        }
    }
    for(size_t i = 0; i < centers.rows(); ++i) {
        VERBOSE_ONLY(std::fprintf(stderr, "center %zu has count %g\n", i, counts[i]);)
        if(!counts[i]) {
            if(!costs) {
                costs.reset(new typename MatrixType::ElementType[nr]);
                OMP_PFOR
                for(size_t j = 0; j < nr; ++j) {
                    //if(j == i) costs[j] = 0.; else
                    costs[j] = func(row(centers, assignments[j]), row(data, j)) * getw(j);
                    //std::fprintf(stderr, "costs[%zu] = %g\n", j, costs[j]);
                }
            }
            std::uniform_real_distribution<typename MatrixType::ElementType> urd;
            ::std::partial_sum(costs.get(), costs.get() + nr, costs.get());
            //for(unsigned i = 0; i < nr; ++i) std::fprintf(stderr, "%u:%g\t", i, costs[i]);
            //std::fputc('\n', stderr);
            size_t item = std::lower_bound(costs.get(), costs.get() + nr, urd(mt)) - costs.get();
            costs[item] = 0.;
            assignments[item] = i;
            //std::fprintf(stderr, "Reassigning center %zu to row %zu because it has lost all support\n", i, item);
            std::fprintf(stderr, "Reassigning center %zu to row %zu because it has lost all support\n", i, item);
            row(centers, i BLAZE_CHECK_DEBUG) = row(data, item);
            centers_reassigned = true;
        }
    }
    if(centers_reassigned)
        goto get_assignment_counts;
    // 2. Assign centers
    double total_loss = 0.;
    OMP_PRAGMA("omp parallel for reduction(+:total_loss)")
    for(size_t i = 0; i < nr; ++i) {
        auto dr = row(data, i BLAZE_CHECK_DEBUG);
        auto lhr = row(centers, 0 BLAZE_CHECK_DEBUG);
        auto dist = blz::serial(func(dr, lhr));
        unsigned label = 0;
        double newdist;
        for(unsigned j = 1;j < centers.rows(); ++j) {
            if((newdist = blz::serial(func(dr, row(centers, j BLAZE_CHECK_DEBUG)))) < dist) {
                //std::fprintf(stderr, "newdist: %g. olddist: %g. Replacing label %u with %u\n", newdist, dist, label, j);
                dist = newdist;
                label = j;
            }
        }
        assignments[i] = label;
        total_loss += getw(i) * dist;
    }
    if(std::isnan(total_loss)) total_loss = std::numeric_limits<decltype(total_loss)>::infinity();
    return total_loss;
}

template<typename IT, typename MatrixType, typename CMatrixType=MatrixType, typename CFT, typename WFT=double,
         typename Functor=blz::sqrL2Norm>
double lloyd_loop(std::vector<IT> &assignments, std::vector<CFT> &counts,
                  CMatrixType &centers, MatrixType &data,
                  double tolerance=0., size_t maxiter=-1,
                  const Functor &func=Functor(),
                  const WFT *weights=static_cast<WFT *>(nullptr),
                  bool use_moving_average=false)
{
    if(tolerance < 0.) throw 1;
    size_t iternum = 0;
    double oldloss = std::numeric_limits<double>::max(), newloss;
    for(;;) {
        newloss = lloyd_iteration(assignments, counts, centers, data, func, weights, use_moving_average);
        double change_in_cost = std::abs(oldloss - newloss) / std::min(oldloss, newloss);
        if(iternum++ == maxiter || change_in_cost <= tolerance) {
            std::fprintf(stderr, "Change in cost from %g to %g is %g\n", oldloss, newloss, change_in_cost);
            break;
        }
        std::fprintf(stderr, "new loss at %zu: %0.30g. old loss: %0.30g\n", iternum, newloss, oldloss);
        oldloss = newloss;
    }
    return newloss;
}

template<typename IT, typename MatrixType, typename CMatrixType=MatrixType, typename WFT=double, typename Functor=blz::sqrL2Norm, typename RNG, typename SelContainer>
void minibatch_lloyd_iteration(std::vector<IT> &assignments, std::vector<WFT> &counts,
                               CMatrixType &centers, MatrixType &data, unsigned batchsize,
                               RNG &rng,
                               SelContainer &selection,
                               const Functor &func=Functor(),
                               const WFT *weights=nullptr)
{
    if(batchsize < assignments.size()) batchsize = assignments.size();
    const size_t np = assignments.size();
    selection.clear();
    selection.reserve(batchsize);
    schism::Schismatic<IT> div(np);
    double weight_sum = 0, dbs = batchsize;
    for(;;) {
        auto ind = div.mod(rng());
        if(std::find(selection.begin(), selection.end(), ind) == selection.end()) {
            blz::push_back(selection, ind);
            if((weight_sum += (weights ? weights[ind]: WFT(1))) >= dbs)
                break;
        }
    }
    shared::sort(selection.begin(), selection.end());
    std::unique_ptr<IT[]> asn(new IT[np]());
    OMP_PRAGMA("omp parallel")
    {
#ifdef _OPENMP
        int nt = omp_get_num_threads();
        std::unique_ptr<std::mutex[]> mutexes(new std::mutex[nt]);
#endif
        std::unique_ptr<IT[]> labels(new IT[batchsize]);
        OMP_PFOR_DYN
        for(size_t i = 0; i < batchsize; ++i) {
            const auto ind = selection[i];
            const auto dr = row(data, ind);
            const auto lhr = row(centers, 0 BLAZE_CHECK_DEBUG);
            double dist = blz::serial(func(dr, lhr)), newdist;
            IT label = 0;
            for(unsigned j = 1; j < centers.rows(); ++j)
                if((newdist = func(dr, row(centers, j BLAZE_CHECK_DEBUG))) < dist)
                    dist = newdist, label = j;
            labels[i] = label;
            OMP_ATOMIC
            counts[label] += weights ? weights[ind]: WFT(1);
        }
        OMP_PFOR_DYN
        for(size_t i = 0; i < batchsize; ++i) {
            const auto label = labels[i];
            const auto ind = selection[i];
            const auto dr = row(data, ind);
            const double eta = (weights ? weights[ind]: WFT(1)) / counts[label];
            auto crow = row(centers, label BLAZE_CHECK_DEBUG);
            OMP_ONLY(std::lock_guard<std::mutex> lock(mutexes[omp_get_thread_num()]);)
            crow = blz::serial((1. - eta) * crow + eta * dr);
        }
    }
}

template<typename IT, typename MatrixType, typename CMatrixType=MatrixType, typename WFT=double,
         typename Functor=blz::sqrL2Norm>
double mb_lloyd_loop(std::vector<IT> &assignments, std::vector<WFT> &counts,
                     CMatrixType &centers, MatrixType &data,
                     unsigned batch_size,
                     size_t maxiter=10000,
                     const Functor &func=Functor(),
                     uint64_t seed=137,
                     const WFT *weights=nullptr)
{
    std::vector<IT> selection;
    selection.reserve(batch_size);
    wy::WyRand<IT, 4> rng(seed);
    size_t iternum = 0;
    while(iternum++ < maxiter) {
        std::fprintf(stderr, "Starting minibatch iter %zu\n", iternum);
        minibatch_lloyd_iteration(assignments, counts, centers, data, batch_size, rng, selection, func, weights);
    }
    double loss = 0.;
    std::memset(counts.data(), 0, sizeof(counts[0]) * counts.size());
    // TODO: Consider moving the centers as well at this step.
    OMP_PRAGMA("omp parallel for reduction(+:loss)")
    for(size_t i = 0; i < assignments.size(); ++i) {
        auto dr = row(data, i BLAZE_CHECK_DEBUG);
        double closs = func(dr, row(centers, 0 BLAZE_CHECK_DEBUG)), newloss;
        IT label = 0;
        for(unsigned j = 1; j < centers.rows(); ++j)
            if((newloss = func(dr, row(centers, j BLAZE_CHECK_DEBUG))) < closs)
                closs = newloss, label = j;
        OMP_ATOMIC
        ++counts[label];
        assignments[i] = label;
        loss += closs;
    }
    return loss;
}



template<typename Iter,
         typename IT=std::uint32_t, typename RNG=wy::WyRand<uint32_t, 2>,
         typename FT=shared::ContainedTypeFromIterator<Iter>, typename Distance=sqrL2Norm>
auto kmeans_coreset(Iter start, Iter end,
                    size_t k, RNG &rng,
                    size_t cs_size,
                    const FT *weights=nullptr, const Distance &dist=Distance()) {
    auto [centers, assignments, sqdists] = kmeanspp(start, end, rng, k, dist);
    using sq_t = typename decltype(sqdists)::value_type;
    coresets::CoresetSampler<sq_t, IT> cs;
    size_t np = end - start;
    cs.make_sampler(np, centers.size(), sqdists.data(), assignments.data(), weights,
                    /*seed=*/rng());
    auto ics(cs.sample(cs_size, rng()));
    DBG_ONLY(for(auto i: ics.indices_) assert(i < np);)
    static_assert(std::is_same<decltype(ics), coresets::IndexCoreset<IT, sq_t>>::value, "must be this type");
    //coresets::IndexCoreset<IT, sq_t> ics(cs.sample(cs_size, rng()));
#ifndef NDEBUG
    std::fprintf(stderr, "max sampled idx: %u\n", *std::max_element(ics.indices_.begin(), ics.indices_.end()));
#endif
    return ics;
}

template<typename MT, bool SO,
         typename IT=std::uint32_t, typename RNG=wy::WyRand<uint32_t, 2>>
auto kmeans_index_coreset(const blaze::Matrix<MT, SO> &mat, size_t k, RNG &rng, size_t cs_size,
                           const blz::ElementType_t<MT> *weights=nullptr, bool rowwise=true)
{
    if(!rowwise) throw std::runtime_error("Not implemented");
    return kmeans_coreset(blz::rowiterator(*mat).begin(), blz::rowiterator(*mat).end(),
                          k, rng, cs_size, weights);
}
template<typename MT, bool SO,
         typename IT=std::uint32_t, typename RNG=wy::WyRand<uint32_t, 2>>
auto kmeans_matrix_coreset(const blaze::Matrix<MT, SO> &mat, size_t k, RNG &rng, size_t cs_size,
                           const blz::ElementType_t<MT> *weights=nullptr, bool rowwise=true)
{
    auto ics = kmeans_index_coreset(mat, k, rng, cs_size, weights, rowwise);
#ifndef NDEBUG
    for(auto idx: ics.indices_)
        assert(idx < rowwise ? (*mat).rows(): (*mat).columns());
    //std::fprintf(stderr, "Got kmeans coreset of size %zu\n", ics.size());
#endif
    return index2matrix(ics, *mat);
}


} // namespace coresets
} // namespace minicore
#endif // FGC_KMEANS_H__
