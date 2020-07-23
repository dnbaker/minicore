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
                             CtrT &costs,
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
    auto cost = compute_cost();
    const auto initcost = cost;
    size_t iternum = 0;
    for(;;) {
        set_centroids_hard<FT>(mat, measure, prior, centers, asn, costs, weights);
        assign_points_hard<FT>(mat, measure, prior, centers, asn, costs, weights);
        auto oldcost = cost;
        cost = compute_cost();
        if(oldcost - cost < eps * initcost || ++iternum == maxiter)
            break;
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
    switch(pol) {
        case NOT_APPLICABLE: throw std::runtime_error("Cannot optimize without a valid centroid policy.");
        case L1_MEDIAN: set_centroids_l1<FT>(~mat, asn, costs, centers, weights);
        case GEO_MEDIAN: set_centroids_l2<FT>(~mat, asn, costs, centers, weights);
        case FULL_WEIGHTED_MEAN: set_centroids_full_mean<FT>(~mat, measure, prior, asn, costs, centers, weights);
        case TVD_MEDIAN: set_centroids_tvd<FT>(~mat, asn, costs, centers, weights);
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
    for(size_t i = 0; i < np; ++i) {
        
    }
    throw NotImplementedError("Not yet");
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
