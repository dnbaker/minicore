#ifndef MINOCORE_CLUSTERING_SOLVE_H__
#define MINOCORE_CLUSTERING_SOLVE_H__
#pragma once

#include "minicore/dist.h"
#include "minicore/clustering/centroid.h"

namespace minicore {

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
                        SumT &centersums,
                        const SumT &rowsums);
template<typename FT, typename Mat, typename PriorT, typename CtrT, typename CostsT, typename AsnT, typename WeightT=CtrT, typename SumT>
void set_centroids_hard(const Mat &mat,
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
         typename SumT>
void set_centroids_soft(const Mat &mat,
                        const dist::DissimilarityMeasure measure,
                        const PriorT &prior,
                        std::vector<CtrT> &centers,
                        CostsT &costs,
                        const WeightT *weights,
                        const FT temp,
                        SumT &centersums,
                        const SumT &rowsums);

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
         typename WeightT=CtrT, // Vector Type
         typename=std::enable_if_t<std::is_floating_point_v<FT>>
        >
auto perform_hard_clustering(const MT &mat, // TODO: consider replacing blaze::Matrix with template Mat for CSR matrices
                             const dist::DissimilarityMeasure measure,
                             const PriorT &prior,
                             std::vector<CtrT> &centers,
                             AsnT &asn,
                             CostsT &costs,
                             const WeightT *weights=static_cast<WeightT *>(nullptr),
                             double eps=1e-10,
                             size_t maxiter=size_t(-1))
{
    auto centers_cpy = centers;
    auto compute_cost = [&costs,w=weights]() -> FT {
        if(w) return blz::dot(costs, *w);
        else  return blz::sum(costs);
    };
#if BLAZE_USE_SHARED_MEMORY_PARALLELIZATION
    const blz::DV<FT> rowsums = sum<blz::rowwise>(mat);
    blz::DV<FT> centersums = blaze::generate(centers.size(), [&](auto x){return blz::sum(centers[x]);});
#else
    blz::DV<FT> rowsums((mat).rows());
    blz::DV<FT> centersums(centers.size());
    OMP_PFOR
    for(size_t i = 0; i < rowsums.size(); ++i)
        rowsums[i] = sum(row(mat, i));
    OMP_PFOR
    for(size_t i = 0; i < centers.size(); ++i)
        centersums[i] = sum(centers[i]);
#endif
    assign_points_hard<FT>(mat, measure, prior, centers, asn, costs, weights, centersums, rowsums); // Assign points myself
    const auto initcost = compute_cost();
    FT cost = initcost;
    std::fprintf(stderr, "initial cost: %0.12g\n", cost);
    size_t iternum = 0;
    for(;;) {
        DBG_ONLY(std::fprintf(stderr, "Beginning iter %zu\n", iternum);)
        set_centroids_hard<FT>(mat, measure, prior, centers_cpy, asn, costs, weights, centersums, rowsums);
        DBG_ONLY(std::fprintf(stderr, "Set centroids %zu\n", iternum);)

        assign_points_hard<FT>(mat, measure, prior, centers_cpy, asn, costs, weights, centersums, rowsums);
        auto newcost = compute_cost();
        std::fprintf(stderr, "Iteration %zu: [%.16g old/%.16g new]\n", iternum, cost, newcost);
        if(newcost > cost) {
            std::cerr << "Warning: New cost " << newcost << " > original cost " << cost << ". Using prior iteration.\n;";
            centersums = blaze::generate(centers.size(), [&](auto x) {return blz::sum(centers[x]);});
            assign_points_hard<FT>(mat, measure, prior, centers, asn, costs, weights, centersums, rowsums);
            //DBG_ONLY(std::abort();)
            break;
        }
        std::swap_ranges(centers.begin(), centers.end(), centers_cpy.begin());
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
#ifndef NDEBUG
    std::fprintf(stderr, "Completing clustering after %zu rounds. Initial cost %0.12g. Final cost %0.12g.\n", iternum, initcost, cost);
#endif
    return std::make_tuple(initcost, cost, iternum);
}


/*
 *
 * set_centroids_hard assumes that costs of points have been assigned
 *
 */
template<typename FT, typename Mat, typename PriorT, typename CtrT, typename CostsT, typename AsnT, typename WeightT=CtrT, typename SumT>
void set_centroids_hard(const Mat &mat,
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
    DBG_ONLY(std::fprintf(stderr, "Policy %d/%s for measure %d/%s\n", (int)pol, cp2str(pol), (int)measure, msr2str(measure));)
    if(dist::is_bregman(measure)) {
        assert(FULL_WEIGHTED_MEAN == pol || JSM_MEDIAN == pol);
    }
    switch(pol) {
        case JSM_MEDIAN:
        case FULL_WEIGHTED_MEAN: set_centroids_full_mean<FT>(mat, measure, prior, asn, costs, centers, weights, ctrsums, rowsums);
            break;
        case L1_MEDIAN:
            set_centroids_l1<FT>(mat, asn, costs, centers, weights);
            break;
        case GEO_MEDIAN:
            set_centroids_l2<FT>(mat, asn, costs, centers, weights);
            break;
#if 0
        case TVD_MEDIAN:
            set_centroids_tvd<FT>(mat, asn, costs, centers, weights);
            break;
#endif
        default:
            constexpr const char *msg = "Cannot optimize without a valid centroid policy.";
            std::cerr << msg;
            throw std::runtime_error(msg);
    }
}

template<typename FT, typename Mat, typename PriorT, typename CtrT, typename CostsT, typename AsnT, typename WeightT=CtrT, typename SumT>
void assign_points_hard(const Mat &mat,
                        const dist::DissimilarityMeasure measure,
                        const PriorT &prior,
                        const std::vector<CtrT> &centers,
                        AsnT &asn,
                        CostsT &costs,
                        const WeightT *,
                        SumT &centersums,
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
        assert(size_t(id) < (*mat).rows());
        auto mr = row(mat, id, blaze::unchecked);
        const auto &ctr = centers[cid];
        const auto rowsum = rowsums[id];
        const auto centersum = centersums[cid];
        assert(ctr.size() == mr.size());
        auto mrmult = mr / rowsum;
        auto wctr = ctr * (1. / centersum);
        //assert(measure == dist::JSD); // Temporary: this is only for sanity checking while debugging JSD calculation
        FT ret;
        switch(measure) {

            // Geometric
            case L1:
                ret = l1Dist(ctr, mr);
            break;
            case L2:    ret = l2Dist(ctr, mr); break;
            case SQRL2: ret = sqrDist(ctr, mr); break;
            case PSL2:  ret = sqrDist(wctr, mrmult); break;
            case PL2:   ret = l2Dist(wctr, mrmult); break;

            // Bregman divergences + convex combinations thereof, and Bhattacharyya
            case HELLINGER:
            case TVD:
            case BHATTACHARYYA_METRIC: case BHATTACHARYYA_DISTANCE:
            case POISSON: case JSD: case JSM:
            case ITAKURA_SAITO: case REVERSE_ITAKURA_SAITO:
            case SIS: case RSIS: case MKL: case UWLLR: case LLR: case SRULRT: case SRLRT:
            case REVERSE_POISSON: case REVERSE_MKL:
                ret = cmp::msr_with_prior(measure, mr, ctr, prior, prior_sum, rowsum, centersum); break;
            case COSINE_DISTANCE:
                ret = dot(mr, ctr) * (1. / (l2Norm(mr) * l2Norm(ctr)));
                break;
            default: {
                const auto msg = std::string("Unsupported measure ") + msr2str(measure) + ", " + std::to_string((int)measure);
                std::cerr << msg;
                throw std::invalid_argument(msg);
            }
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
        } else if(std::isnan(ret)) ret = 0.;
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
         typename FT=blaze::ElementType_t<MT>, // Type of result.
                                               // Defaults to that of MT, but can be parametrized. (e.g., when MT is integral)
         typename CtrT=blz::DynamicVector<FT, rowVector>, // Vector Type
         typename CostsT=blaze::DynamicMatrix<FT, rowMajor>, // Costs matrix, nsamples x ncomponents
         typename PriorT=blaze::DynamicVector<FT, rowVector>,
         typename WeightT=blz::DV<FT, rowVector>, // Vector Type
         typename=std::enable_if_t<std::is_floating_point_v<FT>>
        >
auto perform_soft_clustering(const blaze::Matrix<MT, rowMajor> &mat,
                             const dist::DissimilarityMeasure measure,
                             const PriorT &prior,
                             std::vector<CtrT> &centers,
                             CostsT &costs,
                             FT temperature=1.,
                             const WeightT *weights=static_cast<WeightT *>(nullptr),
                             double eps=1e-40,
                             size_t maxiter=size_t(-1))
{
    auto centers_cpy(centers);
    blz::DV<FT> centersums(centers.size());
    blz::DV<FT> rowsums((*mat).rows());
    std::cerr << "Compute sums\n";
#if BLAZE_USE_SHARED_MEMORY_PARALLELIZATION
    rowsums = blz::sum<blz::rowwise>(*mat);
    centersums = blaze::generate(centers.size(), [&](auto x){return blz::sum(centers[x]);});
#else
    OMP_PFOR
    for(size_t i = 0; i < rowsums.size(); ++i)
        rowsums[i] = blz::sum(row(*mat, i));
    OMP_PFOR
    for(size_t i = 0; i < centers_cpy.size(); ++i)
        centersums[i] = blz::sum(centers_cpy[i]);
#endif
    auto compute_cost = [&]() {
        FT ret = 0.;
#if !BLAZE_USE_SHARED_MEMORY_PARALLELIZATION
        OMP_PRAGMA("omp parallel for reduction(+:ret)")
#endif
        for(size_t i = 0; i < costs.rows(); ++i) {
            auto cr = row(costs, i, blaze::unchecked);
            auto smeval = evaluate(softmax(cr * -temperature));
            FT pointcost;
            if(isnan(smeval)) {
                auto maxind = std::min_element(cr.begin(), cr.end()) - cr.begin();
                smeval.reset();
                smeval[maxind] = 1.;
                pointcost = cr[maxind];
            } else {
                pointcost = dot(smeval, cr);
            }
            if(weights) pointcost *= (*weights)[i];
            ret += pointcost;
        }
        return ret;
    };
    auto cost = compute_cost();
    const auto initcost = cost;
    std::fprintf(stderr, "initial cost: %0.20g\n", cost);
    size_t iternum = 0;
    for(;;) {
        auto oldcost = cost;
        set_centroids_soft<FT>(*mat, measure, prior, centers_cpy, costs, weights, temperature, centersums, rowsums);
        cost = compute_cost();
        std::fprintf(stderr, "oldcost: %.20g. newcost: %.20g. Difference: %0.20g\n", oldcost, cost, oldcost - cost);
        if(oldcost > cost) // Update centers only if an improvement
        {
#if BLAZE_USE_SHARED_MEMORY_PARALLELIZATION
            std::copy(centers_cpy.begin(), centers_cpy.end(), centers.begin());
#else
            OMP_PFOR
            for(unsigned i = 0; i < centers.size(); ++i)
                centers[i] = centers_cpy[i];
#endif
        }
        if(oldcost - cost < eps * initcost || ++iternum == maxiter) {
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
         typename SumT>
void set_centroids_soft(const Mat &mat,
                        const dist::DissimilarityMeasure measure,
                        const PriorT &prior,
                        std::vector<CtrT> &centers,
                        CostsT &costs,
                        const WeightT *weights,
                        const FT temp,
                        SumT &centersums,
                        const SumT &rowsums)
{
    MINOCORE_VALIDATE(dist::is_valid_measure(measure));
    const CentroidPol pol = msr2pol(measure);
    assert(FULL_WEIGHTED_MEAN == pol || !dist::is_bregman(measure) || JSM_MEDIAN == pol); // sanity check
    DBG_ONLY(std::fprintf(stderr, "Policy %d/%s for measure %d/%s\n", (int)pol, cp2str(pol), (int)measure, msr2str(measure));)
    switch(pol) {
        case JSM_MEDIAN:
        case FULL_WEIGHTED_MEAN: set_centroids_full_mean<FT>(mat, measure, prior, costs, centers, weights, temp, centersums);
            break;
        case GEO_MEDIAN: throw NotImplementedError("TODO: implement weighted geometric median from soft clustering. It's just a lot of work.");
        case L1_MEDIAN: throw NotImplementedError("TODO: implement weighted median from soft clustering. It's just a lot of work.");
        default: {
            const std::string msg("Cannot optimize without a valid centroid policy for soft clustering.");
            std::fputs(msg.data(), stderr);
            throw std::runtime_error(msg);
        }
    }
    const FT prior_sum =
        prior.size() == 0 ? 0.
                          : prior.size() == 1
                          ? double(prior[0] * mat.columns())
                          : double(blz::sum(prior));
    auto compute_cost = [&](auto id, auto cid) -> FT {
        auto mr = row(mat, id BLAZE_CHECK_DEBUG);
        const auto rsum = rowsums[id];
        const auto csum = centersums[cid];
        assert(cid < centers.size());
        const auto &ctr = centers[cid];
        assert(ctr.size() == mr.size() || !std::fprintf(stderr, "ctr size: %zu. row size: %zu\n", ctr.size(), mr.size()));
        auto mrmult = mr / rsum;
        auto wctr = ctr / csum;
        FT ret;
        switch(measure) {

            // Geometric
            case L1:
                ret = l1Dist(ctr, mr);
            break;
            case L2:    ret = l2Dist(ctr, mr); break;
            case SQRL2: ret = sqrDist(ctr, mr); break;
            case PSL2:  ret = sqrDist(wctr,  mrmult); break;
            case PL2:   ret = l2Dist(wctr, mrmult); break;

            // Discrete Probability Distribution Measures
            case TVD:       ret = .5 * blz::sum(blz::abs(wctr - mrmult)); break;
            case HELLINGER: ret = l2Norm(blz::sqrt(wctr) - blz::sqrt(mrmult)); break;

            // Bregman divergences, convex combinations thereof, and Bhattacharyya measures
            case BHATTACHARYYA_METRIC: case BHATTACHARYYA_DISTANCE:
            case POISSON: case JSD: case JSM:
            case ITAKURA_SAITO: case REVERSE_ITAKURA_SAITO:
            case SIS: case RSIS: case MKL: case UWLLR: case LLR: case SRULRT: case SRLRT:
            case REVERSE_MKL: case REVERSE_POISSON:
                ret = cmp::msr_with_prior(measure, mr, ctr, prior, prior_sum, rsum, csum); break;
            default: throw NotImplementedError("Unsupported measure for soft clustering");
        }
        return ret;
    };
    costs = blaze::generate(mat.rows(), centers.size(), compute_cost);
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
auto perform_hard_minibatch_clustering(const Matrix &mat, // TODO: consider replacing blaze::Matrix with template Mat for CSR matrices
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
    if(seed == 0) seed = (((uint64_t(std::rand())) << 48) ^ ((uint64_t(std::rand())) << 32)) | ((std::rand() << 16) | std::rand());
#if 0
    auto compute_cost = [&costs,w=weights]() -> FT {
        if(w) return blz::dot(costs, *w);
        else  return blz::sum(costs);
    };
#endif
    switch(measure) {
        default:
        case L1: case TVD: throw std::invalid_argument("measure cannot be used in minibatch mode");

        case JSD: case JSM:
        case SQRL2: case POISSON: case MKL: case REVERSE_ITAKURA_SAITO: case ITAKURA_SAITO:
        case SYMMETRIC_ITAKURA_SAITO: case REVERSE_SYMMETRIC_ITAKURA_SAITO:
        case REVERSE_MKL: case REVERSE_POISSON:
        case LLR: case UWLLR: case SRLRT: case SRULRT:
        case BHATTACHARYYA_METRIC: case BHATTACHARYYA_DISTANCE:
        case HELLINGER:  ; // Do nothing; this should work
    }
#if BLAZE_USE_SHARED_MEMORY_PARALLELIZATION
    const blz::DV<FT> rowsums = sum<blz::rowwise>(mat);
    blz::DV<FT> centersums = blaze::generate(centers.size(), [&](auto x){return blz::sum(centers[x]);});
#else
    blz::DV<FT> rowsums((mat).rows());
    blz::DV<FT> centersums(centers.size());
    OMP_PFOR
    for(size_t i = 0; i < rowsums.size(); ++i)
        rowsums[i] = sum(row(mat, i, blz::unchecked));
    OMP_PFOR
    for(size_t i = 0; i < centers.size(); ++i)
        centersums[i] = blz::sum(centers[i]);
#endif
    FT prior_sum = prior.size() == 1 ? prior.size() * prior[0]: blz::sum(prior);
    size_t iternum = 0;
    double initcost = std::numeric_limits<double>::max(), cost = initcost, bestcost = cost;
    std::vector<CtrT>  savectrs = centers;
    static constexpr FT PI_INV = 1. / 3.14159265358979323846264338327950288;
    using IT = uint64_t;
    auto compute_point_cost = [&](auto id, auto cid) {
        auto mr = row(mat, id, blaze::unchecked);
        const auto &ctr = centers[cid];
        const auto rowsum = rowsums[id];
        const auto centersum = centersums[cid];
        assert(ctr.size() == mr.size());
        //auto mrmult = mr / rowsum;
        //auto wctr = ctr * (1. / centersum);
        //assert(measure == dist::JSD); // Temporary: this is only for sanity checking while debugging JSD calculation
        FT ret;
        switch(measure) {

            case SQRL2: ret = sqrDist(ctr, mr); break;

            // Bregman divergences + convex combinations thereof, and Bhattacharyya
            case TVD:
            case HELLINGER:
            case BHATTACHARYYA_METRIC: case BHATTACHARYYA_DISTANCE:
            case POISSON: case JSD: case JSM:
            case ITAKURA_SAITO: case REVERSE_ITAKURA_SAITO:
            case SIS: case RSIS: case MKL: case UWLLR: case LLR: case SRULRT: case SRLRT:
            case REVERSE_POISSON: case REVERSE_MKL:
            case COSINE_DISTANCE:
                ret = cmp::msr_with_prior(measure, mr, ctr, prior, prior_sum, rowsum, centersum); break;
                break;
            default: {
                const auto msg = std::string("Unsupported measure ") + msr2str(measure) + ", " + std::to_string((int)measure);
                std::cerr << msg;
                throw std::invalid_argument(msg);
            }
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
        } else if(std::isnan(ret)) ret = 0.;
        return ret;
    };
    const size_t np = costs.size(), k = centers.size();
    //shared::flat_hash_map<IT, IT> sa;
    auto perform_assign = [&]() {
        OMP_PFOR_DYN
        for(size_t i = 0; i < np; ++i) {
            FT mincost = std::numeric_limits<FT>::max();
            IT minind = -1;
            for(size_t j = 0; j < k; ++j) {
                const FT nc = compute_point_cost(i, j);
                if(std::isnan(nc)) {
                    std::cerr << i << ", " << j << '\n';
                    throw std::invalid_argument("nan in distance calculation");
                }
                if(nc < mincost) mincost = nc, minind = j;
            }
            asn[i] = minind;
            costs[i] = mincost;
        }
    };
    wy::WyRand<std::make_unsigned_t<IT>> rng(seed);
    schism::Schismatic<std::make_unsigned_t<IT>> div((mat).rows());
    blz::DV<IT> sampled_indices(mbsize);
    //blz::DV<FT> sampled_costs(mbsize);
    blz::DV<FT> center_wsums(k);
    std::vector<std::vector<IT>> assigned(k);
    std::unique_ptr<blz::DV<FT>> wc;
    if(weights) wc.reset(new blz::DV<FT>(np));
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
                OMP_ATOMIC
                ++center_counts[asn[i]];
            }
            blaze::SmallArray<uint32_t, 8> foundindices;
            for(size_t i = 0; i < center_counts.size(); ++i) {
                std::fprintf(stderr, "Center %zu has %" PRIu64 " items\n", i, center_counts[i]);
                if(center_counts[i] <= reseed_after) { // If there are 0 or 1 points assigned to a center, restart it
                    foundindices.pushBack(i);
                }
            }
            if(foundindices.size()) {
                std::fprintf(stderr, "Found %zu centers with no assigned points; let's restart them.\n", foundindices.size());
                for(const auto fidx: foundindices) {
                    // set new centers
                    auto &ctr = centers[fidx];
                    const auto rngv = rng();
                    if(weights) {
                        if constexpr(blaze::IsVector_v<WeightT>) {
                            *wc = costs * *weights;
                        } else if constexpr(std::is_floating_point_v<WeightT>) {
                            *wc = costs * blz::make_cv(weights, np);
                        } else {
                            *wc = costs * blz::make_cv(weights->data(), np);
                        }
                        clustering::set_center(ctr, row(mat, reservoir_simd::sample(wc->data(), np, rngv)));
                    } else {
                        clustering::set_center(ctr, row(mat, reservoir_simd::sample(costs.data(), np, rngv)));
                    }
                    centersums[fidx] = blz::sum(ctr);
                }
                OMP_PFOR
                for(size_t i = 0; i < np; ++i) {
                    auto &ccost = costs[i];
                    for(const auto fidx: foundindices) {
                        auto newcost = compute_point_cost(i, fidx);
                        if(newcost < ccost) ccost = newcost, asn[i] = fidx;
                    }
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
            std::fprintf(stderr, "Cost at iter %zu: %g\n", iternum, cost);
            if(iternum == 0) initcost = cost, bestcost = initcost;
            if(cost < bestcost) {
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
        center_wsums = 0.;
        for(auto &i: assigned) i.clear();
        // 2. Compute nearest centers + step sizes
        OMP_PFOR
        for(size_t i = 0; i < mbsize; ++i) {
            const auto ind = sampled_indices[i];
            IT bestind = -1;
            FT bv = std::numeric_limits<FT>::max();
            for(size_t j = 0; j < k; ++j) {
                auto nv = compute_point_cost(ind, j);
                if(std::isnan(nv)) throw std::invalid_argument(std::string("nan between ") + std::to_string(ind) + " and " + std::to_string(j));
                if(nv < bv)
                    bv = nv, bestind = j;
            }
            //sampled_costs[i] = bv;
            const FT w = weights ? (*weights)[ind]: static_cast<FT>(1);
            OMP_ATOMIC
            center_wsums[bestind] += w;
            OMP_CRITICAL
            {
                assigned[bestind].push_back(ind);
            }
        }
        for(size_t i= 0; i < assigned.size(); ++i) {
            if(!assigned[i].empty()) {
                shared::sort(assigned[i].begin(), assigned[i].end());
            }
        }
        center_wsums = 1. / center_wsums; // center-wsums now contains the eta (step size) for SGD
        // 3. Calculate new center
        OMP_PFOR
        for(size_t i = 0; i < centers.size(); ++i) {
            //const FT eta = center_wsums[i];
            auto asnptr = assigned[i].data();
            const auto asnsz = assigned[i].size();
            if(!asnsz) continue;
            clustering::set_center(centers[i], mat, asnptr, asnsz, weights);
            centersums[i] = sum(centers[i]);
        }
        
        // Set the new centers
        //cost = newcost;
    }
    std::swap(centers, savectrs);
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

} // namespace minicore
#endif /* #ifndef MINOCORE_CLUSTERING_SOLVE_H__ */
