#ifndef MINOCORE_CLUSTERING_SOLVE_H__
#define MINOCORE_CLUSTERING_SOLVE_H__
#pragma once

#include "minocore/dist.h"
#include "minocore/clustering/centroid.h"

namespace minocore {

namespace clustering {


using blz::rowVector;
using blz::columnVector;
using blz::rowMajor;
using blz::columnMajor;

/*
 * set_centroids_* and assign_points_* functions form the E/M steps
 * for EM optimization of clustering
 * See perform_hard_clustering/perform_soft_clustering below for the interface
 */

template<typename FT, typename Mat, typename PriorT, typename CtrT, typename CostsT, typename AsnT, typename WeightT=CtrT>
void set_centroids_hard(const Mat &mat,
                        const dist::DissimilarityMeasure measure,
                        const PriorT &prior,
                        std::vector<CtrT> &centers,
                        AsnT &asn,
                        CostsT &costs,
                        const WeightT *weights=static_cast<WeightT *>(nullptr));
template<typename FT, typename Mat, typename PriorT, typename CtrT, typename CostsT, typename AsnT, typename WeightT=CtrT>
void assign_points_hard(const Mat &mat,
                        const dist::DissimilarityMeasure measure,
                        const PriorT &prior,
                        std::vector<CtrT> &centers,
                        AsnT &asn,
                        CostsT &costs,
                        const WeightT *weights=static_cast<WeightT *>(nullptr));


template<typename FT, typename MT, typename PriorT, typename CtrT, typename CostsT, typename WeightT=CtrT>
void set_centroids_soft(const blaze::Matrix<MT, blz::rowMajor> &mat,
                        const dist::DissimilarityMeasure measure,
                        const PriorT &prior,
                        std::vector<CtrT> &centers,
                        CostsT &costs,
                        const WeightT *weights=static_cast<WeightT *>(nullptr));
template<typename FT, typename MT, typename PriorT, typename CtrT, typename CostsT, typename WeightT=CtrT>
void assign_points_soft(const blaze::Matrix<MT, blz::rowMajor> &mat,
                        const dist::DissimilarityMeasure measure,
                        const PriorT &prior,
                        std::vector<CtrT> &centers,
                        CostsT &costs,
                        const WeightT *weights=static_cast<WeightT *>(nullptr));


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

template<typename MT, // MatrixType
         typename FT=blz::ElementType_t<MT>, // Type of result.
                                             // Defaults to that of MT, but can be parametrized. (e.g., when MT is integral)
         typename CtrT=blz::DynamicVector<FT, rowVector>, // Vector Type
         typename CostsT,
         typename PriorT=blz::DynamicVector<FT, rowVector>,
         typename AsnT=blz::DynamicVector<uint32_t>,
         typename WeightT=CtrT, // Vector Type
         typename=std::enable_if_t<std::is_floating_point_v<FT>>
        >
auto perform_hard_clustering(const blaze::Matrix<MT, blz::rowMajor> &mat, // TODO: consider replacing blaze::Matrix with template Mat for CSR matrices
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
    std::fprintf(stderr, "Beginning perform_hard_clustering with%s weights.\n", weights ? "": "out");
    assign_points_hard<FT>(~mat, measure, prior, centers, asn, costs, weights); // Assign points myself
    const auto initcost = compute_cost();
    FT cost = initcost;
    std::fprintf(stderr, "cost: %0.12g\n", cost);
    size_t iternum = 0;
    for(;;) {
        std::fprintf(stderr, "Beginning iter %zu\n", iternum);
        set_centroids_hard<FT>(~mat, measure, prior, centers, asn, costs, weights);
        std::fprintf(stderr, "Set centroids %zu\n", iternum);

        assign_points_hard<FT>(~mat, measure, prior, centers, asn, costs, weights);
        std::fprintf(stderr, "Assigning points %zu\n", iternum);
        auto newcost = compute_cost();
        std::fprintf(stderr, "Iteration %zu: [%.16g old/%.16g new]\n", iternum, cost, newcost);
        if(newcost > cost) {
            auto msg = std::string("New cost ") + std::to_string(newcost) + " > original cost " + std::to_string(cost) + '\n';
            std::cerr << msg;
            //DBG_ONLY(std::abort();)
            break;
        }
        if(cost - newcost < eps * initcost) {
#ifndef NDEBUG
            std::fprintf(stderr, "Relative cost difference %0.12g compared to threshold %0.12g determined by %0.12g eps and %0.12g init cost\n",
                         cost - newcost, eps * initcost, eps, initcost);
#endif
            break;
        }
        if(++iternum == maxiter) {
#ifndef NDEBUG
            std::fprintf(stderr, "Maximum iterations [%zu] reached\n", iternum);
#endif
            break;
        }
        cost = newcost;
    }
    std::fprintf(stderr, "Completing clustering after %zu rounds. Initial cost %0.12g. Final cost %0.12g.\n", iternum, initcost, cost);
    return std::make_pair(initcost, cost);
}


/*
 *
 * set_centroids_hard assumes that costs of points have been assigned,
 *
 */
template<typename FT, typename Mat, typename PriorT, typename CtrT, typename CostsT, typename AsnT, typename WeightT=CtrT>
void set_centroids_hard(const Mat &mat,
                        const dist::DissimilarityMeasure measure,
                        const PriorT &prior,
                        std::vector<CtrT> &centers,
                        AsnT &asn,
                        CostsT &costs,
                        const WeightT *weights)
{
    MINOCORE_VALIDATE(dist::is_valid_measure(measure));
    const CentroidPol pol = msr2pol(measure);
    std::fprintf(stderr, "Policy %d/%s for measure %d/%s\n", (int)pol, cp2str(pol), (int)measure, msr2str(measure));
    if(dist::is_bregman(measure)) {
        assert(FULL_WEIGHTED_MEAN == pol);
    }
    if(pol == FULL_WEIGHTED_MEAN) set_centroids_full_mean<FT>(mat, measure, prior, asn, costs, centers, weights);
    else if(pol == L1_MEDIAN)            set_centroids_l1<FT>(mat, asn, costs, centers, weights);
    else if(pol == GEO_MEDIAN)           set_centroids_l2<FT>(mat, asn, costs, centers, weights);
    else if(pol == TVD_MEDIAN)          set_centroids_tvd<FT>(mat, asn, costs, centers, weights);
    else throw std::runtime_error("Cannot optimize without a valid centroid policy.");
}

template<typename FT, typename Mat, typename PriorT, typename CtrT, typename CostsT, typename AsnT, typename WeightT=CtrT>
void assign_points_hard(const Mat &mat,
                        const dist::DissimilarityMeasure measure,
                        const PriorT &prior,
                        std::vector<CtrT> &centers,
                        AsnT &asn,
                        CostsT &costs,
                        const WeightT *weights)
{

    // Setup helpers
    // -- Parameters
    using asn_t = std::decay_t<decltype(asn[0])>;
    const size_t np = costs.size();
    const unsigned k = centers.size();
    const FT prior_sum =
        prior.size() == 0 ? 0.
                          : prior.size() == 1
                          ? double(prior[0] * mat.columns())
                          : double(blz::sum(prior));
    blaze::DynamicVector<FT, blaze::rowVector> center_sums = trans(blaze::generate(k, [&centers](auto x) {return blz::sum(centers[x]);}));
    std::fprintf(stderr, "[%s]: %d-clustering with %s and %zu dimensions\n", __func__, k, dist::msr2str(measure), centers[0].size());

    // Compute distance function
    // Handles similarity measure, caching, and the use of a prior for exponential family models
    //
    //
    // TODO: use https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf
    //        triangle inequality to accelerate k-means algorithm
    //       this depends on whether or not a measure is a metric
    //       , or, for the rho-metric generalization
    //       a suitable relaxation allowing similar acceleration.
    //       Also, if there are enough centers, a nearest neighbor structure
    //       could make centroid assignment faster
    auto compute_cost = [&](auto id, auto cid) {
        auto mr = row(mat, id);
        assert(cid < centers.size() || !std::fprintf(stderr, "cid %u, size %zu\n", unsigned(cid), centers.size()));
        const auto &ctr = centers[cid];
        assert(ctr.size() == mr.size());
        auto mrmult = mr / sum(mr);
        auto wctr = ctr * (1. / (center_sums[cid]));
        //assert(measure == dist::JSD); // Temporary: this is only for sanity checking while debugging JSD calculation
#if 0
        std::fprintf(stderr, "Calling compute_cost between item %u and center id %u with measure %d/%s\n",
                     (int)id, (int)cid, (int)measure, dist::msr2str(measure));
#endif
        FT ret;
        switch(measure) {

            // Geometric
            case L1:
                ret = l1Dist(ctr, mr);
                //ret = blz::sum(blz::abs(ctr - mr));
            break; // Replacing l1Norm with blz::sum(blz::abs due to error in norm backend
            case L2:    ret = blz::l2Dist(ctr, mr); break;
            case SQRL2: ret = blz::sqrDist(ctr, mr); break;
            case PSL2:  ret = blz::sqrNorm(wctr - mrmult); break;
            case PL2:   ret = blz::l2Norm(wctr - mrmult); break;

            // Discrete Probability Distribution Measures
            case TVD:       ret = .5 * blz::sum(blz::abs(wctr - mrmult)); break;
            case HELLINGER: ret = blz::l2Norm(blz::sqrt(wctr) - blz::sqrt(mrmult)); break;
            case BHATTACHARYYA_METRIC: case BHATTACHARYYA_DISTANCE: {
                const auto sim = blz::dot(blz::sqrt(wctr), blz::sqrt(mrmult));
                ret = measure == BHATTACHARYYA_METRIC ? std::sqrt(1. - sim)
                                                      : -std::log(sim);
            } break;

            // Bregman divergences + convex combinations thereof
            case POISSON: case JSD:
            case ITAKURA_SAITO: case REVERSE_ITAKURA_SAITO:
            case SIS: case RSIS: case MKL: case UWLLR: case LLR:
                ret = cmp::msr_with_prior(measure, ctr, mr, prior, prior_sum); break;
#if 0
            case PROBABILITY_COSINE_DISTANCE:
                ret = blz::dot(mrmult, wctr) * (1. / (blz::l2Norm(mrmult) * blz::l2Norm(wctr)));
            break;
            case COSINE_DISTANCE:
                ret = blz::dot(mr, ctr) * (1. / (blz::l2Norm(mr) * blz::l2Norm(ctr)));
                break;
            case ORACLE_METRIC: case ORACLE_PSEUDOMETRIC: case COSINE_SIMILARITY: case PROBABILITY_COSINE_SIMILARITY:
            case DOT_PRODUCT_SIMILARITY: case PROBABILITY_DOT_PRODUCT_SIMILARITY:
            case WEMD: case EMD: case OLLR:
            case JSM: std::fprintf(stderr, "No EM algorithm available for measure %d/%s\n", (int)measure, msr2str(measure));
            [[fallthrough]];
#endif
            default: throw std::invalid_argument(std::string("Unsupported measure ") + msr2str(measure));
        }
        if(ret < 0) {
            if(unlikely(ret < -1e-10)) {
                std::fprintf(stderr, "Warning: got a negative distance back %0.12g under %d/%s for ids %u/%u. Check details!\n", ret, (int)measure, msr2str(measure),
                             (unsigned)id, (unsigned)cid);
                std::cerr << ctr << '\n';
                std::cerr << mr << '\n';
                std::abort();
            }
            ret = 0.;
        }
        return ret;
    };
    for(size_t i = 0; i < np; ++i) {
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
template<typename FT, typename MT, typename PriorT, typename CtrT, typename CostsT, typename AsnT, typename WeightT=CtrT>
void assign_points_soft(const blaze::Matrix<MT, blz::rowMajor> &mat,
                        const dist::DissimilarityMeasure measure,
                        const PriorT &prior,
                        std::vector<CtrT> &centers,
                        AsnT &asn,
                        CostsT &costs,
                        const WeightT *weights)
{
#if 0
    using asn_t = std::decay_t<decltype(asn[0])>;
    const size_t np = costs.size();
    const unsigned k = centers.size();
    for(size_t i = 0; i < np; ++i) {
        auto cost = compute_cost(i, 0);
        asn_t bestid = 0;
        for(unsigned j = 1; j < k; ++j)
            if(auto newcost = compute_cost(i, 1); newcost < cost)
                bestid = j, cost = newcost;
        costs[i] = cost;
        asn[i] = bestid;
    }
#else
    throw TODOError("Not completed");
#endif
}

#if 0
template<typename MT, // MatrixType
         typename FT=blaze::ElementType_t<MT>, // Type of result.
                                               // Defaults to that of MT, but can be parametrized. (e.g., when MT is integral)
         typename CtrT=blz::DynamicVector<FT, rowVector>, // Vector Type
         typename CostsT=blaze::DynamicMatrix<FT, rowMajor>, // Costs matrix, nsamples x ncomponents
         typename PriorT=blaze::DynamicVector<FT, rowVector>,
         typename AsnT=blaze::DynamicVector<uint32_t>,
         typename WeightT=CtrT, // Vector Type
         typename=std::enable_if_t<std::is_floating_point_v<FT>>
        >
auto perform_soft_clustering(const blaze::Matrix<MT, rowMajor> &mat,
                             const dist::DissimilarityMeasure measure,
                             const PriorT &prior,
                             std::vector<CtrT> &centers,
                             CostsT &costs,
                             const WeightT *weights=static_cast<WeightT *>(nullptr),
                             double eps=1e-10,
                             size_t maxiter=size_t(-1))
{
    auto compute_cost = [&]() {
        FT ret;
        if(weights)
            ret = blaze::sum(blaze::softmax<blaze::rowwise>(costs) % costs % blz::expand(*weights, costs.columns()));
        else
            ret = blaze::sum(blaze::softmax<blaze::rowwise>(costs) % costs);
        return ret;
    };
    auto cost = compute_cost();
    const int k = centers.size();
    const auto initcost = cost;
    size_t iternum = 0;
    for(;;) {
        auto oldcost = cost;
        set_centroids_soft<FT>(mat, measure, prior, centers, costs, weights);
        assign_points_soft<FT>(mat, measure, prior, centers, costs, weights);
        cost = compute_cost();
        if(oldcost - cost < eps * initcost || ++iternum == maxiter)
            break;
    }
    return std::make_pair(initcost, cost);
}
#endif



} // namespace clustering

} // namespace minocore
#endif /* #ifndef MINOCORE_CLUSTERING_SOLVE_H__ */
