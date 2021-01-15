#ifndef MINOCORE_CLUSTERING_SOLVE_H__
#define MINOCORE_CLUSTERING_SOLVE_H__
#pragma once

#include "minicore/dist.h"
#include "minicore/clustering/centroid.h"
#include "minicore/coreset/coreset.h"

namespace minicore {

namespace clustering {


using blz::rowVector;
using blz::columnVector;
using blz::rowMajor;
using blz::columnMajor;
using blz::unchecked;

using coresets::l1_median;
using util::l1_median;
using coresets::tvd_median;
using util::tvd_median;

/*
 * set_centroids_* and assign_points_* functions form the E/M steps
 * for EM optimization of clustering
 * See perform_hard_clustering/perform_soft_clustering below for the interface
 */

/*
 *
 * This is the be used after acquiring an approximate solution
 * and storing it in centers, costs in costs, and assignments in asn
 *
 * Estimation
 * 1. The algorithm begins by checking for any centers with 0 points assigned,
 * and "restarts" such points using D^2 sampling.
 * 2. Then, it calculates centroids for each such cluster. (Maximization)
 * Maximization
 * 3. Then, assign points to their nearest centers.
 * While not converged, continue
 */
template<typename FT, typename Mat, typename PriorT, typename CtrT, typename CostsT, typename AsnT, typename WeightT=CtrT, typename SumT>
void assign_points_hard(const Mat &mat,
                        const dist::DissimilarityMeasure measure,
                        const PriorT &prior,
                        const std::vector<CtrT> &centers,
                        AsnT &asn,
                        CostsT &costs,
                        const WeightT *,
                        const SumT &centersums,
                        const SumT &rowsums);
template<typename FT, typename Mat, typename PriorT, typename CtrT, typename CostsT, typename AsnT, typename WeightT=CtrT, typename SumT>
bool set_centroids_hard(const Mat &mat,
                        const dist::DissimilarityMeasure measure,
                        const PriorT &prior,
                        std::vector<CtrT> &centers,
                        AsnT &asn,
                        CostsT &costs,
                        const WeightT *weights,
                        SumT &ctrsums,
                        const SumT &);
template<typename FT, typename Mat, typename PriorT, typename CtrT,
         typename CostsT,
         typename WeightT,
         typename SumT, typename RSumT>
double set_centroids_soft(const Mat &mat,
                        const dist::DissimilarityMeasure measure,
                        const PriorT &prior,
                        std::vector<CtrT> &centers,
                        CostsT &costs,
                        CostsT &asns,
                        const WeightT *weights,
                        const FT temp,
                        SumT &centersums,
                        const RSumT &rowsums);

template<typename MT>
using DefaultFT = std::conditional_t<std::is_floating_point_v<blz::ElementType_t<MT>>,
                                                              blz::ElementType_t<MT>,
                                                              std::conditional_t<sizeof(blz::ElementType_t<MT>) < 8, float, double>>;

template<typename MT, // MatrixType
         typename FT=DefaultFT<MT>,
         typename CtrT=blz::DynamicVector<FT, rowVector>, // Vector Type
         typename CostsT,
         typename PriorT=blz::DynamicVector<FT, rowVector>,
         typename AsnT=blz::DynamicVector<uint32_t>,
         typename WeightT=blz::DynamicVector<FT>, // Vector Type
         typename=std::enable_if_t<std::is_floating_point_v<FT>>
        >
std::tuple<double, double, size_t>
perform_hard_clustering(const MT &mat,
                        const dist::DissimilarityMeasure measure,
                        const PriorT &prior,
                        std::vector<CtrT> &centers,
                        AsnT &asn,
                        CostsT &costs,
                        const WeightT *weights=static_cast<WeightT *>(nullptr),
                        double eps=1e-10,
                        size_t maxiter=size_t(-1))
{
    auto compute_cost = [&costs,w=weights]() -> FT {
        if(w) return blz::dot(costs, *w);
        else  return blz::sum(costs);
    };
    const blz::DV<FT> rowsums = sum<blz::rowwise>(mat);
    blz::DV<FT> centersums = blaze::generate(centers.size(), [&](auto x){return sum(centers[x]);});
    assign_points_hard<FT>(mat, measure, prior, centers, asn, costs, weights, centersums, rowsums); // Assign points myself
    const auto initcost = compute_cost();
    FT cost = initcost;
    std::fprintf(stderr, "[%s] initial cost: %0.12g\n", __PRETTY_FUNCTION__, cost);
    if(cost == 0) {
        std::fprintf(stderr, "Cost is 0 (unexpected), but the cost can't decrease. No optimization performed\n");
        return {0., 0., 0};
    }
    size_t iternum = 0;
    auto centers_cpy = centers;
    for(;;) {
        std::fprintf(stderr, "Beginning iter %zu\n", iternum);
        auto res = set_centroids_hard<FT>(mat, measure, prior, centers_cpy, asn, costs, weights, centersums, rowsums);
        //std::fprintf(stderr, "Set centroids %zu\n", iternum);

        assign_points_hard<FT>(mat, measure, prior, centers_cpy, asn, costs, weights, centersums, rowsums);
        auto newcost = compute_cost();
        std::fprintf(stderr, "Iteration %zu: [%.16g old/%.16g new]\n", iternum, cost, newcost);
        if(newcost > cost && !res) {
            std::cerr << "Warning: New cost " << newcost << " > original cost " << cost << ". Using prior iteration.\n;";
            centersums = blaze::generate(centers.size(), [&](auto x) {return sum(centers[x]);});
            assign_points_hard<FT>(mat, measure, prior, centers, asn, costs, weights, centersums, rowsums);
            break;
        }
        centers = centers_cpy;
        if(cost - newcost < eps * initcost) {
#ifndef NDEBUG
            std::fprintf(stderr, "Relative cost difference %0.12g compared to threshold %0.12g determined by %0.12g eps and %0.12g init cost\n",
                         cost - newcost, eps * initcost, eps, initcost);
#endif
            break;
        }
        if(++iternum == maxiter) {
            std::fprintf(stderr, "Maximum iterations [%zu] reached\n", iternum);
            break;
        }
        cost = newcost;
    }
#ifndef NDEBUG
    std::fprintf(stderr, "Completing clustering after %zu rounds. Initial cost %0.12g. Final cost %0.12g.\n", iternum, initcost, cost);
#endif
    return {initcost, cost, iternum};
}


/*
 *
 * set_centroids_hard assumes that costs of points have been assigned
 * Returns True if a center was restarted; for this case, we don't force termination of
 * the clustering algorithm
 */
template<typename FT, typename Mat, typename PriorT, typename CtrT, typename CostsT, typename AsnT, typename WeightT=blz::DV<FT>, typename SumT>
bool set_centroids_hard(const Mat &mat,
                        const dist::DissimilarityMeasure measure,
                        const PriorT &prior,
                        std::vector<CtrT> &centers,
                        AsnT &asn,
                        CostsT &costs,
                        const WeightT *weights,
                        SumT &ctrsums,
                        const SumT &rowsums)
{
    MINOCORE_VALIDATE(dist::is_valid_measure(measure));
    const CentroidPol pol = msr2pol(measure);
    if(dist::is_bregman(measure)) {
        assert(FULL_WEIGHTED_MEAN == pol || JSM_MEDIAN == pol);
    }
    bool ctrs_restarted = false;
    switch(pol) {
        case JSM_MEDIAN:
        case FULL_WEIGHTED_MEAN: ctrs_restarted |= set_centroids_full_mean<FT>(mat, measure, prior, asn, costs, centers, weights, ctrsums, rowsums);
            break;
        case L1_MEDIAN:
            set_centroids_l1<FT>(mat, asn, costs, centers, weights);
            break;
        case GEO_MEDIAN:
            set_centroids_l2<FT>(mat, asn, costs, centers, weights);
            break;
        case TVD_MEDIAN:
            set_centroids_tvd<FT>(mat, asn, costs, centers, weights, rowsums);
            break;
        default:
            constexpr const char *msg = "Cannot optimize without a valid centroid policy.";
            std::cerr << msg;
            throw std::runtime_error(msg);
    }
    for(size_t i = 0; i < centers.size(); ++i) {
        ctrsums[i] = sum(centers[i]);
        //std::fprintf(stderr, "After setting, ctr %zu has %g for a sum\n", i, ctrsums[i]);
    }
    return ctrs_restarted;
}

template<typename FT, typename Mat, typename PriorT, typename CtrT, typename CostsT, typename AsnT, typename WeightT=CtrT, typename SumT>
void assign_points_hard(const Mat &mat,
                        const dist::DissimilarityMeasure measure,
                        const PriorT &prior,
                        const std::vector<CtrT> &centers,
                        AsnT &asn,
                        CostsT &costs,
                        const WeightT *,
                        const SumT &centersums,
                        const SumT &rowsums)
{

    // Setup helpers
    // -- Parameters
    using asn_t = std::decay_t<decltype(asn[0])>;
    const FT prior_sum =
        prior.size() == 0 ? 0.
                          : prior.size() == 1
                          ? double(prior[0] * mat.columns())
                          : double(blz::sum(prior));
    assert(centersums.size() == centers.size());
    assert(rowsums.size() == (*mat).rows());
#ifndef NDEBUG
    std::fprintf(stderr, "[%s]: %zu-clustering with %s and %zu dimensions\n", __func__, centers.size(), dist::msr2str(measure), centers[0].size());
#endif

    // Compute distance function
    // Handles similarity measure, caching, and the use of a prior for exponential family models
    auto compute_cost = [&](auto id, auto cid) ALWAYS_INLINE {
        FT ret = cmp::msr_with_prior<FT>(measure, row(mat, id), centers[cid], prior, prior_sum, rowsums[id], centersums[cid]);
        if(ret < 0) {
            if(unlikely(ret < -1e-5)) {
                std::fprintf(stderr, "rowsum: %g. csum: %g. expected rsum: %g expected csum: %g\n", double(sum(row(mat, id))), double(sum(centers[cid])), rowsums[id], centersums[cid]);
                std::fprintf(stderr, "Warning: got a negative distance back %0.12g under %d/%s for ids %u/%u. Check details. Total L1 distance: %g\n", ret, (int)measure, msr2str(measure),
                             (unsigned)id, (unsigned)cid, l1Dist(centers[cid], row(mat, id)));
                std::cerr << centers[cid] << '\n';
                std::cerr << row(mat, id) << '\n';
                std::abort();
            }
            ret = 0.;
        } else if(std::isnan(ret)) {
            std::fprintf(stderr, "ret: %g\n", ret);
            throw std::runtime_error("nan");
            ret = 0.;
        }
        return ret;
    };
    const size_t e = costs.size(), k = centers.size();
    OMP_PFOR
    for(size_t i = 0; i < e; ++i) {
        auto cost = compute_cost(i, 0);
        asn_t bestid = 0;
        for(unsigned j = 1; j < k; ++j)
            if(auto newcost = compute_cost(i, j); newcost < cost)
                bestid = j, cost = newcost;
        costs[i] = cost;
        asn[i] = bestid;
        VERBOSE_ONLY(std::fprintf(stderr, "point %zu is assigned to center %u with cost %0.12g\n", i, bestid, cost);)
    }
}

template<typename MT, // MatrixType
         typename FT=std::conditional_t<(sizeof(ElementType_t<MT>) <= 4), float, double>,
         typename CtrT=blz::DynamicVector<FT, rowVector>, // Vector Type
         typename CostsT=blaze::DynamicMatrix<FT, rowMajor>, // Costs matrix, nsamples x ncomponents
         typename PriorT=blaze::DynamicVector<FT, rowVector>,
         typename WeightT=blz::DV<FT, rowVector>, // Vector Type
         typename=std::enable_if_t<std::is_floating_point_v<FT>>
        >
auto perform_soft_clustering(const MT &mat,
                             const dist::DissimilarityMeasure measure,
                             const PriorT &prior,
                             std::vector<CtrT> &centers,
                             CostsT &costs,
                             CostsT &asns,
                             double temperature=1.,
                             size_t maxiter=size_t(-1),
                             int64_t mbsize=-1, int64_t mbn=10,
                             const WeightT *weights=static_cast<WeightT *>(nullptr),
                             double eps=0.)
{
    auto centers_cpy(centers);
    blz::DV<FT> centersums(centers.size());
    blz::DV<FT> rowsums((mat).rows());
    rowsums = sum<rowwise>(mat);
    centersums = blaze::generate(centers.size(), [&](auto x){return blz::sum(centers[x]);});
    double cost = std::numeric_limits<double>::max();
    double initcost = -1;
    size_t iternum = 0;
    for(;;) {
        double oldcost = cost;
        if(mbsize > 0) {
            throw std::runtime_error("Not yet completed: minibatch soft clustering");
            for(int i = 0; i < mbn; ++i); // Perform mbn rounds of minibatch clustering between central
        } else {
            cost = set_centroids_soft<FT>(mat, measure, prior, centers_cpy, costs, asns, weights, temperature, centersums, rowsums);
        }
        if(initcost < 0) {
            initcost = cost;
            std::fprintf(stderr, "[%s] initial cost: %0.12g\n", __PRETTY_FUNCTION__, cost);
        }
        DBG_ONLY(std::fprintf(stderr, "oldcost: %.20g. newcost: %.20g. Difference: %0.20g\n", oldcost, cost, oldcost - cost);)
        if(oldcost >= cost) // Update centers only if an improvement
            std::copy(centers_cpy.begin(), centers_cpy.end(), centers.begin());
        if(oldcost - cost <= eps * initcost || ++iternum == maxiter) {
            break;
        }
    }
    return std::make_tuple(initcost, cost, iternum);
}
/*
 *
 * set_centroids_soft assumes that costs of points have been assigned
 *
 */
template<typename FT, typename Mat, typename PriorT, typename CtrT,
         typename CostsT,
         typename WeightT,
         typename SumT, typename RSumT>
double set_centroids_soft(const Mat &mat,
                        const dist::DissimilarityMeasure measure,
                        const PriorT &prior,
                        std::vector<CtrT> &centers,
                        CostsT &costs,
                        CostsT &asns,
                        const WeightT *weights,
                        const FT temp,
                        SumT &centersums,
                        const RSumT &rowsums)
{
    MINOCORE_VALIDATE(dist::is_valid_measure(measure));
    const CentroidPol pol = msr2pol(measure);
    assert(FULL_WEIGHTED_MEAN == pol || !dist::is_bregman(measure) || JSM_MEDIAN == pol); // sanity check
    std::fprintf(stderr, "Policy %d/%s for measure %d/%s\n", (int)pol, cp2str(pol), (int)measure, msr2str(measure));
    double ret = set_centroids_full_mean(mat, measure, prior, costs, asns, centers, weights, temp, centersums, rowsums);
    std::fprintf(stderr, "cost: %g for %d/%s\n", ret, (int)measure, msr2str(measure));
    const double prior_sum =
        prior.size() == 0 ? 0.
                          : prior.size() == 1
                          ? double(prior[0] * mat.columns())
                          : double(blz::sum(prior));
    costs = blaze::generate(mat.rows(), centers.size(), [&](auto id, auto cid) ALWAYS_INLINE {
        assert(cid < centers.size());
        return cmp::msr_with_prior<FT>(measure, row(mat, id, unchecked), centers[cid], prior, prior_sum, rowsums[id], centersums[cid]);
    });
    //std::cerr << "Costs: " << costs << '\n';
    return ret;
}


template<typename Matrix, // MatrixType
         typename FT=DefaultFT<Matrix>,
         typename CtrT=blz::DynamicVector<FT, rowVector>, // Vector Type
         typename CostsT,
         typename PriorT=blz::DynamicVector<FT, rowVector>,
         typename AsnT=blz::DynamicVector<uint32_t>,
         typename WeightT=CtrT, // Vector Type
         typename=std::enable_if_t<std::is_floating_point_v<FT>>
        >
auto perform_hard_minibatch_clustering(const Matrix &mat,
                                       const dist::DissimilarityMeasure measure,
                                       const PriorT &prior,
                                       std::vector<CtrT> &centers,
                                       AsnT &asn,
                                       CostsT &costs,
                                       const WeightT *weights=static_cast<WeightT *>(nullptr),
                                       size_t mbsize=1000,
                                       size_t maxiter=10000,
                                       size_t calc_cost_freq=100,
                                       unsigned int reseed_after=1,
                                       bool with_replacement=true,
                                       uint64_t seed=0)
{
    const bool isnorm = msr_is_normalized(measure);
    if(seed == 0) seed = (((uint64_t(std::rand())) << 48) ^ ((uint64_t(std::rand())) << 32)) | ((std::rand() << 16) | std::rand());
    const blz::DV<double> rowsums = sum<blz::rowwise>(mat);
    blz::DV<double> centersums = blaze::generate(centers.size(), [&](auto x){return blz::sum(centers[x]);});
    const double prior_sum = prior.size() == 1 ? prior.size() * prior[0]: blz::sum(prior);
    size_t iternum = 0;
    double initcost = std::numeric_limits<double>::max(), cost = initcost, bestcost = cost;
    std::vector<CtrT>  savectrs = centers;
    using IT = uint64_t;
    auto compute_point_cost = [&](auto id, auto cid) ALWAYS_INLINE {
        FT ret = cmp::msr_with_prior<FT>(measure, row(mat, id, unchecked), centers[cid], prior, prior_sum, rowsums[id], centersums[cid]);
        if(ret < 0 || std::isnan(ret))
            ret = 0.;
        else if(std::isinf(ret))
            ret = std::numeric_limits<FT>::max(); // To make it finite
        return ret;
    };
    const size_t np = costs.size(), k = centers.size();
    auto perform_assign = [&]() {
        OMP_PFOR_DYN
        for(size_t i = 0; i < np; ++i) {
            FT mincost = std::numeric_limits<FT>::max();
            IT minind = -1;
            for(size_t j = 0; j < k; ++j)
                if(const FT nc = compute_point_cost(i, j);nc < mincost)
                    mincost = nc, minind = j;
            asn[i] = minind;
            costs[i] = mincost;
        }
    };
    wy::WyRand<std::make_unsigned_t<IT>> rng(seed);
    schism::Schismatic<std::make_unsigned_t<IT>> div((mat).rows());
    blz::DV<IT> sampled_indices(mbsize);
    std::vector<std::vector<IT>> assigned(k);
    blz::DV<FT> wc;
    if(weights) wc.resize(np);
    blz::DV<uint64_t> center_counts(k);
    for(;;) {
        DBG_ONLY(std::fprintf(stderr, "Beginning iter %zu\n", iternum);)
        if(iternum % calc_cost_freq == 0 || (iternum == maxiter - 1)) {
            // Every once in a while, perform exhaustive center-point-comparisons
            // and restart any centers with no assigned points
            perform_assign();
            center_counts = 0;

            OMP_PFOR
            for(size_t i = 0; i < np; ++i) {
                assert(asn[i] < k);
                OMP_ATOMIC
                ++center_counts[asn[i]];
            }
            blaze::SmallArray<uint32_t, 8> foundindices;
            for(size_t i = 0; i < center_counts.size(); ++i)
                if(center_counts[i] <= reseed_after) // If there are few points assigned to a center, restart it
                    foundindices.pushBack(i);
            if(foundindices.size()) {
                DBG_ONLY(std::fprintf(stderr, "Found %zu centers with no assigned points; restart them.\n", foundindices.size());)
                for(const auto fidx: foundindices) {
                    // set new centers
                    auto &ctr = centers[fidx];
                    size_t id;
                    if(weights) {
                        if constexpr(blaze::IsVector_v<WeightT>) {
                            wc = costs * *weights;
                        } else if constexpr(std::is_floating_point_v<WeightT>) {
                            wc = costs * blz::make_cv(weights, np);
                        } else {
                            wc = costs * blz::make_cv(weights->data(), np);
                        }
                        id = reservoir_simd::sample(wc.data(), np, rng());
                    } else {
                        id = reservoir_simd::sample(costs.data(), np, rng());
                    }
                    if(isnorm) clustering::set_center(ctr, row(mat, id, blz::unchecked) / rowsums[id]);
                    else       clustering::set_center(ctr, row(mat, id, blz::unchecked));
                    centersums[fidx] = sum(ctr);
                }
                OMP_PFOR
                for(size_t i = 0; i < np; ++i) {
                    auto &ccost = costs[i];
                    for(const auto fidx: foundindices)
                        if(auto newcost = compute_point_cost(i, fidx);newcost < ccost)
                             ccost = newcost, asn[i] = fidx;
                }
            }
            if(weights) {
                if constexpr(blaze::IsVector_v<WeightT>) {
                    cost = blz::dot(costs, *weights);
                } else {
                    cost = blz::dot(costs, blz::make_cv(weights->data(), np));
                }
            } else {
                cost = blz::sum(costs);
            }
            std::fprintf(stderr, "Cost at iter %zu (mbsize %zd): %g\n", iternum, mbsize, cost);
            if(iternum == 0) initcost = cost, bestcost = initcost;
            if(cost < bestcost) {
                std::fprintf(stderr, "Distance between old and new centers: %g\n", blz::sum(blz::generate(centers.size(), [&](auto x) {return l2Dist(centers[x], savectrs[x]);})));
                bestcost = cost;
                savectrs = centers;
            }
        }

        if(++iternum == maxiter) {
            std::fprintf(stderr, "Maximum iterations [%zu] reached\n", iternum);
            break;
        }

        // 1. Sample the points
        if(with_replacement) {
            SK_UNROLL_8
            for(size_t i = 0; i < mbsize; ++i) {
                sampled_indices[i] = div.mod(rng());
            }
        } else {
            shared::flat_hash_set<IT> selidx;
            for(;selidx.size() < mbsize; selidx.insert(div.mod(rng())));
            std::copy(selidx.begin(), selidx.end(), sampled_indices.data());
        }
        for(auto &i: assigned) i.clear();
        OMP_ONLY(auto locks = std::make_unique<std::mutex[]>(k);)
        // 2. Compute nearest centers + step sizes
        OMP_PFOR
        for(size_t i = 0; i < mbsize; ++i) {
            const auto ind = sampled_indices[i];
            IT oldasn = asn[ind], bestind = -1;
            FT bv = std::numeric_limits<FT>::max();
            for(size_t j = 0; j < k; ++j)
                if(auto nv = compute_point_cost(ind, j); nv < bv)
                    bv = nv, bestind = j;
            if(bestind == (IT(-1)))
                bestind = oldasn;
            {
                OMP_ONLY(std::lock_guard<std::mutex> lock(locks[bestind]);)
                assigned[bestind].push_back(ind);
            }
        }
        OMP_PFOR
        for(size_t i= 0; i < assigned.size(); ++i) {
            shared::sort(assigned[i].begin(), assigned[i].end());
        }
        // 3. Calculate new center
        OMP_PFOR
        for(size_t i = 0; i < centers.size(); ++i) {
            //const FT eta = center_wsums[i];
            auto asnptr = assigned[i].data();
            const auto asnsz = assigned[i].size();
            if(!asnsz) continue;
            if(measure == distance::L2) {
                clustering::set_center_l2(centers[i], mat, asnptr, asnsz, weights);
            } else if(measure == distance::L1) {
                l1_median(mat, centers[i], asnptr, asnsz, weights);
            } else if(measure == distance::TVD) {
                tvd_median(mat, centers[i], asnptr, asnsz, weights, rowsums);
            } else {
                clustering::set_center(centers[i], mat, asnptr, asnsz, weights);
            }
            //std::cerr << "Center[" << i << "] " << trans(centers[i]) << '\n';
            VERBOSE_ONLY(std::cerr << "##center with sum " << sum(centers[i]) << " and index "  << i << ": " << centers[i] << '\n';)
            centersums[i] = sum(centers[i]);
            VERBOSE_ONLY(std::fprintf(stderr, "center sum: %g. csums: %g\n", centersums[i], sum(centers[i]));)
        }
        // Set the new centers
        //cost = newcost;
    }
    centers = savectrs;
    cost = bestcost;
#ifndef NDEBUG
    std::fprintf(stderr, "Completing clustering after %zu rounds. Initial cost %0.12g. Final cost %0.12g.\n", iternum, initcost, cost);
#endif
    return std::make_tuple(initcost, cost, iternum);
}
// hard minibatch coreset clustering
template<typename Matrix, // MatrixType
         typename FT=DefaultFT<Matrix>,
         typename CtrT=blz::DynamicVector<FT, rowVector>, // Vector Type
         typename CostsT,
         typename PriorT=blz::DynamicVector<FT, rowVector>,
         typename AsnT=blz::DynamicVector<uint32_t>,
         typename WeightT=CtrT, // Vector Type
         typename=std::enable_if_t<std::is_floating_point_v<FT>>
        >
auto hmb_coreset_clustering(const Matrix &mat,
                            const dist::DissimilarityMeasure measure,
                            const PriorT &prior,
                            std::vector<CtrT> &centers,
                            AsnT &asn,
                            CostsT &costs,
                            const WeightT *weights=static_cast<WeightT *>(nullptr),
                            size_t mbsize=1000,
                            size_t maxiter=10000,
                            size_t calc_cost_freq=100,
                            unsigned int reseed_after=1,
                            uint64_t seed=0)
{
    const bool isnorm = msr_is_normalized(measure);
    if(seed == 0) seed = (((uint64_t(std::rand())) << 48) ^ ((uint64_t(std::rand())) << 32)) | ((std::rand() << 16) | std::rand());
    const blz::DV<double> rowsums = sum<blz::rowwise>(mat);
    blz::DV<double> centersums = blaze::generate(centers.size(), [&](auto x){return blz::sum(centers[x]);});
    const double prior_sum = prior.size() == 1 ? prior.size() * prior[0]: blz::sum(prior);
    size_t iternum = 0;
    double initcost = std::numeric_limits<double>::max(), cost = initcost, bestcost = cost;
    std::vector<CtrT>  savectrs = centers;
    using IT = uint64_t;
    auto compute_point_cost = [&](auto id, auto cid) ALWAYS_INLINE {
        FT ret = cmp::msr_with_prior<FT>(measure, row(mat, id, unchecked), centers[cid], prior, prior_sum, rowsums[id], centersums[cid]);
        if(ret < 0 || std::isnan(ret))
            ret = 0.;
        else if(std::isinf(ret))
            ret = std::numeric_limits<FT>::max(); // To make it finite
        return ret;
    };
    const size_t np = costs.size(), k = centers.size();
    wy::WyRand<std::make_unsigned_t<IT>> rng(seed);
    schism::Schismatic<std::make_unsigned_t<IT>> div((mat).rows());
    blz::DV<IT> sampled_indices(mbsize);
    std::vector<std::vector<IT>> assigned(k);
    blz::DV<FT> wc;
    if(weights) wc.resize(np);
    blz::DV<uint64_t> center_counts(k);
    for(;;) {
        DBG_ONLY(std::fprintf(stderr, "Beginning iter %zu\n", iternum);)
        if(1) {
            // Every once in a while, perform exhaustive center-point-comparisons
            // and restart any centers with no assigned points
            OMP_PFOR_DYN
            for(size_t i = 0; i < np; ++i) {
                FT mincost = std::numeric_limits<FT>::max();
                IT minind = -1;
                for(size_t j = 0; j < k; ++j)
                    if(const FT nc = compute_point_cost(i, j);nc < mincost)
                        mincost = nc, minind = j;
                asn[i] = minind;
                costs[i] = mincost;
            }
            center_counts = 0;

            OMP_PFOR
            for(size_t i = 0; i < np; ++i) {
                assert(asn[i] < k);
                OMP_ATOMIC
                ++center_counts[asn[i]];
            }
            blaze::SmallArray<uint32_t, 8> foundindices;
            for(size_t i = 0; i < center_counts.size(); ++i)
                if(center_counts[i] <= reseed_after) // If there are few points assigned to a center, restart it
                    foundindices.pushBack(i);
            if(foundindices.size()) {
                DBG_ONLY(std::fprintf(stderr, "Found %zu centers with no assigned points; restart them.\n", foundindices.size());)
                for(const auto fidx: foundindices) {
                    // set new centers
                    auto &ctr = centers[fidx];
                    size_t id;
                    if(weights) {
                        if constexpr(blaze::IsVector_v<WeightT>) {
                            wc = costs * *weights;
                        } else if constexpr(std::is_floating_point_v<WeightT>) {
                            wc = costs * blz::make_cv(weights, np);
                        } else {
                            wc = costs * blz::make_cv(weights->data(), np);
                        }
                        id = reservoir_simd::sample(wc.data(), np, rng());
                    } else {
                        id = reservoir_simd::sample(costs.data(), np, rng());
                    }
                    if(isnorm) clustering::set_center(ctr, row(mat, id, blz::unchecked) / rowsums[id]);
                    else       clustering::set_center(ctr, row(mat, id, blz::unchecked));
                    centersums[fidx] = sum(ctr);
                }
                OMP_PFOR
                for(size_t i = 0; i < np; ++i) {
                    auto &ccost = costs[i];
                    for(const auto fidx: foundindices)
                        if(auto newcost = compute_point_cost(i, fidx);newcost < ccost)
                             ccost = newcost, asn[i] = fidx;
                }
            }
            if(weights) {
                if constexpr(blaze::IsVector_v<WeightT>)
                    cost = blz::dot(costs, *weights);
                else cost = blz::dot(costs, blz::make_cv(weights->data(), np));
            } else cost = blz::sum(costs);
            std::fprintf(stderr, "[CSOPT] Cost at iter %zu (mbsize %zd): %g. [best prev: %g]\n", iternum, mbsize, cost, bestcost);
            if(iternum == 0) initcost = cost, bestcost = initcost;
            if(cost < bestcost) {
                std::fprintf(stderr, "[CSOPT] at iter %zu, new cost %g is better than previous %g\n", iternum, cost, bestcost);
                std::fprintf(stderr, "dist between: %g\n", blz::sum(blz::generate(centers.size(), [&](auto x) {return l2Dist(centers[x], savectrs[x]);})));
                bestcost = cost;
                savectrs = centers;
            }
        }

        if(++iternum >= maxiter) {
            std::fprintf(stderr, "[CSOPT] Maximum iterations [%zu] reached\n", maxiter);
            break;
        }

        // Sample points
        auto sampler = coresets::CoresetSampler();
        const coresets::SensitivityMethod sm = measure == L1 || measure == L2 ? coresets::VX: coresets::LBK;
        using WT = const std::remove_const_t<std::decay_t<decltype((*weights)[0])>>;
        const WT *ptr = nullptr;
        if(weights) ptr = weights->data();
        sampler.make_sampler(np, k, costs.data(), asn.data(), ptr, sm);
        for(size_t j = 0; j < calc_cost_freq; ++j) {
            auto pts = sampler.sample(mbsize, rng());
            pts.compact();
            using LMat = std::conditional_t<blaze::IsDenseMatrix_v<Matrix>,
                                blz::DM<blz::ElementType_t<Matrix>>,
                                blz::SM<blz::ElementType_t<Matrix>>
            >;
            LMat smat(pts.size(), mat.columns());
            const size_t tnz = sum(blaze::generate(pts.size(), [&](auto x) {return nonZeros(row(mat, pts.indices_[x]));}));
            smat.reserve(tnz);
            OMP_PFOR
            for(size_t i = 0; i < pts.indices_.size(); ++i) {
                auto lh = row(smat, i);
                set_center(lh, row(mat, pts.indices_[i]));
            }
                //row(smat, i) = row(mat, pts.indices_[i]);
            blz::DV<double> csw;
            if(weights)
                csw = elements(*weights, pts.indices_.data(), pts.indices_.size()) * pts.weights_;
            else
                csw = pts.weights_;
            blz::DV<double> cscosts(pts.size());
            blz::DV<uint32_t> csasn(pts.size());
            OMP_PFOR
            for(size_t i = 0; i < pts.size(); ++i) {
                auto rs = rowsums[pts.indices_[i]];
                auto sr = row(smat, i, unchecked);
                auto bestscore = cmp::msr_with_prior<FT>(measure, sr, centers[0], prior, prior_sum, rs, centersums[0]);
                auto bestid = 0u;
                for(size_t j = 1; j < k; ++j) {
                    auto nscore = cmp::msr_with_prior<FT>(measure, sr, centers[j], prior, prior_sum, rs, centersums[j]);
                    if(nscore < bestscore) bestid = j, bestscore = nscore;
                }
                csasn[i] = bestid;
                cscosts[i] = bestscore;
            }
            perform_hard_clustering(smat, measure, prior, centers, csasn, cscosts, &csw);
            centersums = blaze::generate(centers.size(), [&](auto x) {return sum(centers[x]);});
            ++iternum;
        }

        // 2. Optimize over the coreset
        // 3. Calculate new centers
        // Set the new centers
    }
    centers = savectrs;
    cost = bestcost;
#ifndef NDEBUG
    std::fprintf(stderr, "Completing clustering after %zu rounds. Initial cost %0.12g. Final cost %0.12g.\n", iternum, initcost, cost);
#endif
    return std::make_tuple(initcost, cost, iternum);
}



} // namespace clustering
using clustering::perform_hard_clustering;
using clustering::perform_hard_minibatch_clustering;
using clustering::perform_soft_clustering;
using clustering::hmb_coreset_clustering;

} // namespace minicore
#endif /* #ifndef MINOCORE_CLUSTERING_SOLVE_H__ */
