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

template<typename FT, typename MT, typename PriorT, typename CtrT, typename CostsT, typename AsnT, typename WeightT=CtrT>
void set_centroids_hard(const blaze::Matrix<MT, blz::rowMajor> &mat,
                        const dist::DissimilarityMeasure measure,
                        const PriorT &prior,
                        std::vector<CtrT> &centers,
                        AsnT &asn,
                        CostsT &costs,
                        const WeightT *weights=static_cast<WeightT *>(nullptr));
template<typename FT, typename MT, typename PriorT, typename CtrT, typename CostsT, typename AsnT, typename WeightT=CtrT>
void assign_points_hard(const blaze::Matrix<MT, blz::rowMajor> &mat,
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
auto perform_hard_clustering(const blaze::Matrix<MT, blz::rowMajor> &mat,
                             const dist::DissimilarityMeasure measure,
                             const PriorT &prior,
                             std::vector<CtrT> &centers,
                             AsnT &asn,
                             CostsT &costs,
                             const WeightT *weights=static_cast<WeightT *>(nullptr),
                             double eps=1e-10,
                             size_t maxiter=size_t(-1))
{
    auto compute_cost = [&]() {
        FT ret;
        if(weights)
            ret = blz::dot(costs, *weights);
        else
            ret = blz::sum(costs);
        return ret;
    };
    const int k = centers.size();
    const size_t np = costs.size();
    std::fprintf(stderr, "Beginning perform_hard_clustering %s weights.\n", weights ? " with": " without");
    auto cost = compute_cost();
    std::fprintf(stderr, "cost: %g\n", cost);
    const auto initcost = cost;
    size_t iternum = 0;
    for(;;) {
        std::fprintf(stderr, "Beginning iter %zu\n", iternum);
        set_centroids_hard<FT>(mat, measure, prior, centers, asn, costs, weights);
        std::fprintf(stderr, "Set centroids %zu\n", iternum);
        
        assign_points_hard<FT>(mat, measure, prior, centers, asn, costs, weights);
        std::fprintf(stderr, "Assigning points %zu\n", iternum);
        auto newcost = compute_cost();
        if(cost - newcost < eps * initcost || ++iternum == maxiter)
            break;
        cost = newcost;
    }
    return std::make_pair(initcost, cost);
}


/*
 *
 * set_centroids_hard assumes that costs of points have been assigned,
 *
 */
template<typename FT, typename MT, typename PriorT, typename CtrT, typename CostsT, typename AsnT, typename WeightT=CtrT>
void set_centroids_hard(const blaze::Matrix<MT, blz::rowMajor> &mat,
                        const dist::DissimilarityMeasure measure,
                        const PriorT &prior,
                        std::vector<CtrT> &centers,
                        AsnT &asn,
                        CostsT &costs,
                        const WeightT *weights)
{
    MINOCORE_VALIDATE(dist::is_valid_measure(measure));
    const CentroidPol pol = msr2pol(measure);
    if(dist::is_bregman(measure)) {
        assert(FULL_WEIGHTED_MEAN == pol);
    }
    if(pol == FULL_WEIGHTED_MEAN) set_centroids_full_mean<FT>(~mat, measure, prior, asn, costs, centers, weights);
    else if(pol == L1_MEDIAN)            set_centroids_l1<FT>(~mat, asn, costs, centers, weights);
    else if(pol == GEO_MEDIAN)           set_centroids_l2<FT>(~mat, asn, costs, centers, weights);
    else if(pol == TVD_MEDIAN)          set_centroids_tvd<FT>(~mat, asn, costs, centers, weights);
    else throw std::runtime_error("Cannot optimize without a valid centroid policy.");
}

#if 0
    cached value helpers
    // TODO: partition out cached quantities
    // into an abstraction which is shared by all iterates
    // and not re-computed each time
    // -- Cached values
    //    row sums -- multiply by inverse using SIMD rather than use division
    //    square root: only for Hellinger/Bhattacharyya
    std::unique_ptr<CtrT[]> srcenters;
    // Inverse sums for normalization
    std::unique_ptr<double[]> icsums(new double[k]);
    blz::DV<double, blz::rowVector> irowsums = 1. / trans(blz::sum<blz::rowwise>(~mat));
    blz::DV<double, blz::rowVector> l2centers, l2points;
    for(unsigned i = 0; i < k; ++i) icsums[i] = 1. / blz::sum(centers[i]);
    if(dist::needs_sqrt(measure)) {
        srcenters.reset(new CtrT[k]);
        for(unsigned i = 0; i < k; srcenters[i] = blz::sqrt(centers[i] / blz::sum(centers[i])), ++i);
    }
    if(dist::needs_l2_cache(measure) || dist::needs_probability_l2_cache(measure)) {
        l2centers.resize(k);
        l2points.resize(np);
        for(unsigned i = 0; i < k; ++i) {
            if(dist::needs_l2_cache(measure))
                l2centers[i] = 1. /  blz::l2Norm(centers[i]);
            else
                l2centers[i] = 1. /  blz::l2Norm(centers[i] * icsums[i]);
        }
        for(unsigned i = 0; i < np; ++i) {
            if(dist::needs_l2_cache(measure))
                l2points[i] = 1. /  blz::l2Norm(row(~mat, i));
            else
                l2points[i] = 1. /  blz::l2Norm(row(~mat, i) * irowsums[i]);
        }
    }

#endif

template<typename FT=double, typename CtrT, typename MatrixRowT, typename PriorT>
FT msr_with_prior(dist::DissimilarityMeasure msr, const CtrT &ctr, const MatrixRowT &mr, const PriorT &prior, double prior_sum)
{
    if(!blaze::IsSparseVector_v<CtrT> || !blaze::IsSparseVector_v<MatrixRowT> || prior_sum == 0.) {
        auto logr = blz::neginf2zero(blz::log(mr));
        auto logc = blz::neginf2zero(blz::log(ctr));
        switch(msr) {
            default: throw TODOError("Not yet done");
            case JSM: case JSD: {
                FT ret;
                auto mn = .5 * (mr + ctr);
                auto lmn = blaze::neginf2zero(log(mn));
                ret = .5 * (blz::dot(mr, logr - lmn) + blz::dot(ctr, logc - lmn));
                if(msr == JSM) ret = std::sqrt(ret);
                return ret;
            }
            case MKL: return blz::dot(mr, logr - logc);
        }
    } else {
        const size_t nd = mr.size();
        auto perform_core = [&](auto &src, auto &ctr, auto init, const auto &sharedfunc, const auto &lhofunc, const auto &rhofunc, const auto &nsharedfunc)
            -> FT
                ALWAYS_INLINE
        {
            if constexpr(blaze::IsSparseVector_v<std::decay_t<decltype(src)>> && blaze::IsSparseVector_v<std::decay_t<decltype(ctr)>>) {
                const size_t sharednz = merge::for_each_by_case(nd,
                                        src.begin(), src.end(), ctr.begin(), ctr.end(),
                                        [&](auto, auto x, auto y) {init += sharedfunc(x, y);},
                                        [&](auto, auto x) {init += lhofunc(x);},
                                        [&](auto, auto y) {init += rhofunc(y);});
                init += nsharedfunc(sharednz);
            } else if constexpr(blaze::IsDenseVector_v<std::decay_t<decltype(src)>> && blaze::IsDenseVector_v<std::decay_t<decltype(ctr)>>) {
                throw TODOError("");
            } else {
                throw TODOError("mixed densities;");
            }
            return init;
        };
        // Perform core now takes:
        // 1. Initialization
        // 2-4. Functions for sharednz, lhnz, rhnz
        // 5. Function for number of shared zeros
        // This template allows us to concisely describe all of the exponential family models + convex combinations thereof we support
        FT ret;
        const FT lhsum = blz::sum(mr) + prior_sum;
        const FT rhsum = blz::sum(ctr) + prior_sum;
        const FT lhrsi = 1. / lhsum, rhrsi = 1. / rhsum; // TODO: cache sums?
        const FT lhinc = prior[0] * lhrsi, rhinc = prior[0] * rhrsi;
        const FT rhl = std::log(rhinc), rhincl = rhl * rhinc;
        const FT lhl = std::log(lhinc), lhincl = lhl * lhinc;
        const FT shl = std::log((lhinc + rhinc) * .5), shincl = (lhinc + rhinc) * .5 * shl;;
        auto wr = mr * lhrsi;  // wr and wc are weighted/normalized centers/rows
        auto wc = ctr * rhrsi; //
        // TODO: consider batching logs from sparse vectors with some extra dispatching code
        auto __isc = [&](auto x) ALWAYS_INLINE {return x - std::log(x);};
        // Consider -ffast-math/-fassociative-math
        switch(msr) {
            case JSM:
            case JSD: {
                ret = perform_core(wr, wc, FT(0),
                   [&](auto xval, auto yval) ALWAYS_INLINE {
                        auto xv = xval + lhinc, yv = yval + rhinc;
                        auto addv = xv + yv, halfv = addv * .5;
                        return .5 * (xv * std::log(xv) + yv * std::log(yv) - std::log(halfv) * addv);
                    },
                    /* xonly */    [&](auto xval) ALWAYS_INLINE  {
                        auto xv = xval + lhinc;
                        auto addv = xv + rhinc, halfv = addv * .5;
                        return .5 * (xv * std::log(xv) + rhincl - std::log(halfv) * addv);
                    },
                    /* yonly */    [&](auto yval) ALWAYS_INLINE  {
                        auto yv = yval + rhinc;
                        auto addv = yv + lhinc, halfv = addv * .5;
                        return .5 * (yv * std::log(yv) + lhincl - std::log(halfv) * addv);
                    },
                    /*sharedz*/    [mult=(lhincl + rhincl - shincl) * .5](auto x) {
                        return x * mult;
                    });
                if(msr == JSM) ret = std::sqrt(ret);
            }
            case ITAKURA_SAITO: {
                ret = perform_core(wr, wc, -FT(nd),
                    /* shared */   [&](auto xval, auto yval) ALWAYS_INLINE {
                        return __isc((xval + lhinc) / (yval + rhinc));
                    },
                    /* xonly */    [&](auto xval) ALWAYS_INLINE  {return __isc((xval + lhinc) * rhrsi);},
                    /* yonly */    [&](auto yval) ALWAYS_INLINE  {return __isc(lhinc / (yval + rhinc));},
                    /*sharedz*/    [&,mult=__isc(rhsum * lhrsi)](auto x) {return x * mult;});
            }
            case REVERSE_ITAKURA_SAITO:
                ret = perform_core(wr, wc, -FT(nd),
                    /* shared */   [&](auto xval, auto yval) ALWAYS_INLINE {
                        return __isc((yval + rhinc) / (xval + lhinc));
                    },
                    /* xonly */    [&](auto xval) ALWAYS_INLINE  {return __isc(rhinc / (xval + lhinc));},
                    /* yonly */    [&](auto yval) ALWAYS_INLINE  {return __isc(lhrsi * (yval + rhinc));},
                    /*sharedz*/    [&,mult=__isc(lhsum * rhrsi)](auto x) {return x * mult;});
            case MKL: {
                ret = perform_core(wr, wc, 0.,
                    /* shared */   [&](auto xval, auto yval) ALWAYS_INLINE {return (xval + lhinc) * (std::log((xval + lhinc) / (yval + rhinc)));},
                    /* xonly */    [&](auto xval) ALWAYS_INLINE  {return (xval + lhinc) * (std::log(xval + lhinc) + rhl);},
                    /* yonly */    [&](auto yval) ALWAYS_INLINE  {return lhinc * std::log(yval + rhinc);},
                    /*sharedz*/    [lr=-lhinc * rhl](auto x) {return lr * x;});
            }
            break;
            case SIS:
            case RSIS:
            case UWLLR: case LLR:
            default: throw TODOError("unexpected msr; not yet supported");
        }
        return ret;
    }
}

template<typename FT, typename MT, typename PriorT, typename CtrT, typename CostsT, typename AsnT, typename WeightT=CtrT>
void assign_points_hard(const blaze::Matrix<MT, blz::rowMajor> &mat,
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
                          ? double(prior[0] * (~mat).columns())
                          : double(blz::sum(prior));

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
        auto mr = row(~mat, id, blaze::unchecked);
        const auto &ctr = centers[cid];
#if 1
        auto mrmult = mr / blz::sum(mr);
        auto wctr = ctr / blz::sum(ctr);
#else
        auto mrmult = mr * irowsums[id];
        auto wctr = ctr * icsums[cid];
#endif
        FT ret;
        switch(measure) {

            // Geometric
            case L1: ret = blz::l1Norm(ctr - mr); break;
            case L2: ret = blz::l2Norm(ctr - mr); break;
            case SQRL2: ret = blz::sqrNorm(ctr - mr); break;
            case PSL2: ret = blz::sqrNorm(wctr - mrmult); break;
            case PL2: ret = blz::l2Norm(wctr - mrmult); break;

            // Discrete Probability Distribution Measures
            case TVD: ret = .5 * blz::sum(blz::abs(wctr - mrmult)); break;
            case HELLINGER: ret = blz::l2Norm(blz::sqrt(wctr) - blz::sqrt(mrmult)); break;
            case BHATTACHARYYA_METRIC: case BHATTACHARYYA_DISTANCE: {
                const auto sim = blz::dot(blz::sqrt(wctr), blz::sqrt(mrmult));
                ret = measure == BHATTACHARYYA_METRIC ? std::sqrt(1. - sim)
                                                      : -std::log(sim);
            } break;

            case POISSON: case JSD:
            case ITAKURA_SAITO: case REVERSE_ITAKURA_SAITO:
            case SIS: case RSIS: case MKL:
                ret = msr_with_prior(measure, ctr, mr, prior, prior_sum); break;
            case PROBABILITY_COSINE_DISTANCE:
#if 1
                ret = blz::dot(mrmult, wctr) * (1. / (blz::l2Norm(mrmult) * blz::l2Norm(wctr)));
#else
                ret = blz::dot(mrmult, wctr) * l2points[id] * l2centers[cid];
#endif
            break;
            case COSINE_DISTANCE:
#if 1
                ret = blz::dot(mr, ctr) * (1. / (blz::l2Norm(mr) * blz::l2Norm(ctr)));
#else
                ret = blz::dot(mr, ctr) * l2points[id] * l2centers[cid];
#endif
                break;
            case ORACLE_METRIC: case ORACLE_PSEUDOMETRIC: case COSINE_SIMILARITY: case PROBABILITY_COSINE_SIMILARITY:
            case DOT_PRODUCT_SIMILARITY: case PROBABILITY_DOT_PRODUCT_SIMILARITY:
            case WEMD: case EMD: case OLLR:
            case JSM: std::fprintf(stderr, "No EM algorithm available for measure %d/%s\n", (int)measure, msr2str(measure));
            [[fallthrough]];
            default: throw std::invalid_argument(std::string("Unupported measure ") + msr2str(measure));
        }
        return ret;
    };
    for(size_t i = 0; i < np; ++i) {
        auto cost = compute_cost(i, 0);
        asn_t bestid = 0;
        for(unsigned j = 1; j < k; ++j)
            if(auto newcost = compute_cost(i, 1); newcost < cost)
                bestid = j, cost = newcost;
        costs[i] = cost;
        asn[i] = bestid;
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
