#ifndef FGC_CLUSTERING_DISPATCH_H__
#define FGC_CLUSTERING_DISPATCH_H__
#include "minocore/dist.h"
#include "minocore/optim/jv_solver.h"
#include "minocore/optim/lsearch.h"
#include "minocore/optim/oracle_thorup.h"
#include "minocore/util/exception.h"
#include "diskmat/diskmat.h"
#include "minocore/clustering/traits.h"
#include "minocore/clustering/sampling.h"
#include "minocore/clustering/centroid.h"
#include <cstdint>

namespace minocore {

namespace clustering {

using dist::DissimilarityMeasure;
using blaze::ElementType_t;
using diskmat::PolymorphicMat;

template<typename T>
bool use_packed_distmat(const T &app) {
    if constexpr(jsd::is_dissimilarity_applicator_v<T>) {
        return dist::detail::is_symmetric(app.get_measure());
    }
    return true;
}

template<typename IT=uint32_t, typename FT, typename WFT=FT, typename OracleType, typename Traits>
auto perform_cluster_metric_kmedian(const OracleType &app, size_t np, Traits traits)
{
    MetricSelectionResult<IT, FT> ret;

    std::unique_ptr<dm::DistanceMatrix<FT, 0, dm::DM_MMAP>> distmatp;
    std::unique_ptr<PolymorphicMat<FT>> full_distmatp;
    if(traits.compute_full) {
        if(use_packed_distmat(app)) {
            distmatp.reset(new dm::DistanceMatrix<FT, 0, dm::DM_MMAP>(np));
            for(size_t i = 0; i < np; ++i) {
                auto [ptr, extent] = distmatp->row_span(i);
                const auto offset = i + 1;
                OMP_PFOR_DYN
                for(size_t j = 0; j < extent; ++j)
                    ptr[j] = app(i, j + offset);
            }
        } else {
            full_distmatp.reset(new PolymorphicMat<FT>(np, np));
            for(size_t i = 0; i < np; ++i) {
                auto r = row(full_distmatp->operator~(), i, blaze::unchecked);
                OMP_PFOR_DYN
                for(size_t j = 0; j < np; ++j) {
                    r[j] = app(i, j);
                }
            }
        }
    }
    auto fill_distance_mat = [&](const auto &lu) {
        auto &retdm = std::get<3>(ret);
        retdm.resize(std::get<0>(ret).size(), np);
        for(size_t i = 0; i < std::get<0>(ret).size(); ++i) {
            const auto cid = std::get<0>(ret)[i];
            auto rowptr = row(retdm, i);
            OMP_PFOR
            for(size_t j = 0; j < np; ++j) {
                rowptr[j] = (unlikely(j == cid) ? static_cast<FT>(0.): FT(lu(cid, j)));
            }
        }
    };
    if(traits.sampling == THORUP_SAMPLING) {
        auto sample_and_fill = [&](const auto &x) {
           std::tie(std::get<0>(ret), std::get<1>(ret), std::get<2>(ret))
                = iterated_oracle_thorup_d(
               x, np, traits.k, traits.thorup_iter, traits.thorup_sub_iter, traits.weights, traits.thorup_npermult, 3, 0.5, traits.seed);
            fill_distance_mat(x);
        };
        if(distmatp) {
            sample_and_fill(*distmatp);
        } else if(full_distmatp) {
            sample_and_fill(~*full_distmatp);
        } else {
            auto caching_app = make_row_caching_oracle_wrapper<
                shared::flat_hash_map, /*is_symmetric=*/ true, /*is_threadsafe=*/true
            >(app, np);
            sample_and_fill(caching_app);
        }
    } else switch(traits.sampling) {
        case D2_SAMPLING: {
            ret = select_d2(app, np, traits);
            break;
        }
        case UNIFORM_SAMPLING: {
            ret = select_uniform_random(app, np, traits);
            break;
        }
        case GREEDY_SAMPLING: {
            ret = select_greedy(app, np, traits);
            break;
        }
        case DEFAULT_SAMPLING: default: {
            char buf[128];
            auto l = std::sprintf(buf, "Unrecognized sampling: %d\n", (int)DEFAULT_SAMPLING);
            throw std::invalid_argument(std::string(buf, l));
        }
        fill_distance_mat(app);
    }
    auto &costmat = ret.facility_cost_matrix();
    std::vector<IT> center_sol;
    switch(traits.metric_solver) {
        case JAIN_VAZIRANI_FL: case JV_PLUS_LOCAL_SEARCH: {
            auto jvs = jv::make_jv_solver(costmat);
            auto [c_centers, c_assignments] = jvs.kmedian(traits.k, traits.max_jv_rounds);
            if(traits.metric_solver == JAIN_VAZIRANI_FL) {
                center_sol = std::move(c_centers);
                break;
            }
            // JV_PLUS_LOCAL_SEARCH
            auto lsearcher = minocore::make_kmed_lsearcher(costmat, traits.k, traits.eps, traits.seed);
            lsearcher.lazy_eval_ = 2;
            lsearcher.assign_centers(c_centers.begin(), c_centers.end());
            lsearcher.run();
            center_sol.assign(lsearcher.sol_.begin(), lsearcher.sol_.end());
            break;
        }
        case LOCAL_SEARCH: {
            auto lsearcher = minocore::make_kmed_lsearcher(costmat, traits.k, traits.eps, traits.seed);
            lsearcher.lazy_eval_ = 2;
            lsearcher.run();
            center_sol.assign(lsearcher.sol_.begin(), lsearcher.sol_.end());
            break;
        }
        default: throw std::invalid_argument("Unrecognized metric solver strategy");
    }
    std::transform(center_sol.begin(), center_sol.end(), center_sol.begin(),
                   [&sel=ret.selected()](auto x) {return sel[x];});
    blaze::DynamicVector<IT> asn(np, center_sol.front());
    blaze::DynamicVector<FT> costs = trans(row(costmat, center_sol.front()));
    for(unsigned ci = 1; ci < center_sol.size(); ++ci) {
        auto r = row(costmat, ci);
        OMP_PFOR
        for(size_t i = 0; i < np; ++i) {
            if(auto newv = r[i]; newv < costs[i])
                costs[i] = newv, asn[i] = center_sol[ci];
        }
    }
    shared::sort(center_sol.begin(), center_sol.end());
    return std::make_tuple(center_sol, asn, costs);
}

enum LloydLoopResult {
    FINISHED,
    REACHED_MAX_ROUNDS,
    UNFINISHED
};

template<Assignment asn_method=HARD, CenterOrigination co=EXTRINSIC, typename MatrixType, typename CentersType, typename Assignments, typename WFT=ElementType_t<MatrixType>,
         typename CostType>
LloydLoopResult perform_lloyd_loop(CentersType &centers, Assignments &assignments,
    const jsd::DissimilarityApplicator<MatrixType> &app,
    unsigned k, CostType &retcost, uint64_t seed=0, const WFT *weights=static_cast<WFT *>(nullptr),
    size_t max_iter=100, double eps=1e-4)
{
    if(co != EXTRINSIC) throw std::invalid_argument("Must be extrinsic for Lloyd's");
    using FT = ElementType_t<MatrixType>;
    auto &mat = app.data();
    CentersType centers_cpy(centers), centers_cache;
    MINOCORE_REQUIRE(centers.size() == k, "Must have the correct number of centers");
    if(dist::detail::needs_logs(app.get_measure()) || dist::detail::needs_sqrt(app.get_measure()))
        centers_cache.resize(centers.size());
    double last_distance = std::numeric_limits<double>::max(), first_distance = last_distance,
           center_distance;
    LloydLoopResult ret = UNFINISHED;
    wy::WyRand<uint64_t> rng(seed);
    size_t iternum = 0;
    auto get_center_change_distance = [&]() {
        center_distance = std::accumulate(centers_cpy.begin(), centers_cpy.end(), 0.,
            [&](double value, auto &center) {
                auto ind = std::distance(&centers_cpy.front(), &center);
                return value + blaze::sum(blaze::abs(center - centers[ind]));
            }
        );
        std::swap(centers_cpy, centers);
        if(last_distance == std::numeric_limits<double>::max()) {
            last_distance = first_distance = center_distance;
            iternum = 1;
        } else {
            last_distance = center_distance;
            if(center_distance / first_distance < eps)
                ret = LloydLoopResult::FINISHED;
            else if(++iternum > max_iter)
                ret = LloydLoopResult::REACHED_MAX_ROUNDS;
        }
    };
    using cv_t = blaze::CustomVector<WFT, blaze::unaligned, blaze::unpadded, blaze::rowVector>;
    std::unique_ptr<cv_t> weight_cv;
    if(weights) {
        weight_cv.reset(new cv_t(const_cast<WFT *>(weights), app.size()));
    }
    auto getcache = [&] (size_t j) {
        return centers_cache.size() ? &centers_cache[j]: static_cast<decltype(&centers_cache[j])>(nullptr);
    };
    if constexpr(asn_method == HARD) {
        std::vector<std::vector<uint32_t>> assigned(centers.size());
        OMP_ONLY(std::unique_ptr<std::mutex[]> mutexes(new std::mutex[centers.size()]);)
        for(;;) {
            // Do it forever
            if(centers_cache.size()) {
                for(size_t i = 0; i < centers.size(); ++i)
                    dist::detail::set_cache(centers[i], centers_cache[i], app.get_measure());
            }
            for(auto &i: assigned) i.clear();
            OMP_PFOR
            for(size_t i = 0; i < app.size(); ++i) {
                auto dist = app(i, centers[0], getcache(0));
                unsigned asn = 0;
                for(size_t j = 1; j < centers.size(); ++j) {
                    auto newdist = app(i, centers[j], getcache(j));
                    if(newdist < dist) {
                        asn = j;
                        dist = newdist;
                    }
                }
                assignments[i] = asn;
                {
                    OMP_ONLY(std::unique_lock<std::mutex> lock(mutexes[asn]);)
                    assigned[asn].push_back(i);
                }
            }
            std::vector<size_t> centers_to_restart;
            for(size_t i = 0; i < assignments.size(); ++i) {
                if(assigned[i].empty()) {
                    centers_to_restart.push_back(i);
                }
            }
            if(centers_to_restart.size()) {
                // Use D^2 sampling to start a new cluster
                // And then restart the loop
                blaze::DynamicVector<FT> ccosts(assignments.size(), std::numeric_limits<FT>::max());
                OMP_PFOR
                for(size_t i = 0; i < app.size(); ++i) {
                    for(size_t j = 0; j < centers.size(); ++j) {
                        auto fc = app(i, centers[j], getcache(j));
                        if(fc < ccosts[i]) ccosts[i] = fc;
                    }
                }
                blaze::DynamicVector<FT> csum(assignments.size());
                std::partial_sum(ccosts.data(), ccosts.data() + ccosts.size(), csum.data());
                std::uniform_real_distribution<FT> urd;
                auto nc = centers_to_restart.size();
                for(size_t i = 0; i < nc; ++i) {
                    auto cid = centers_to_restart[i];
                    auto newp = std::lower_bound(csum.data(), csum.data() + csum.size(), urd(rng) * csum[csum.size() - 1]) - csum.data();
                    centers[cid] = row(app.data(), newp, blaze::unchecked);
                    OMP_PFOR
                    for(size_t i = 0; i < assignments.size(); ++i) {
                        auto fc = app(i, newp);
                        if(fc < ccosts[i]) ccosts[i] = fc;
                    }
                    std::partial_sum(ccosts.data(), ccosts.data() + ccosts.size(), csum.data());
                }
                continue; // Reassign, re-center, and re-compute
            }
            // Make assignments
            for(size_t i = 0; i < centers_cpy.size(); ++i) {
                auto &cref = centers_cpy[i];
                auto &assigned_ids = assigned[i];
                shared::sort(assigned_ids.begin(), assigned_ids.end()); // Better access pattern
                if(weight_cv) {
                    auto wsel = blaze::elements(*weight_cv, assigned_ids.data(), assigned_ids.size());
                    CentroidPolicy::perform_average(
                        cref,
                        rows(mat, assigned_ids.data(), assigned_ids.size()),
                        blaze::elements(app.row_sums(), assigned_ids.data(), assigned_ids.size()),
                        &wsel, app.get_measure()
                    );
                } else {
                    using ptr_t = std::add_pointer_t<decltype(blaze::elements(*weight_cv, assigned_ids.data(), assigned_ids.size()))>;
                    CentroidPolicy::perform_average(
                        cref,
                        rows(mat, assigned_ids.data(), assigned_ids.size()),
                        blaze::elements(app.row_sums(), assigned_ids.data(), assigned_ids.size()),
                        static_cast<ptr_t>(nullptr), app.get_measure()
                    );
                }
            }
            get_center_change_distance();
            if(ret != UNFINISHED) goto end;
        }
        // Set the returned values to be the last iteration's.
    } else {
        const size_t nc = centers.size(), nr = app.size();
        if(assignments.rows() != app.size() || assignments.columns() != centers.size()) {
            assignments.resize(app.size(), centers.size());
        }
        std::unique_ptr<std::mutex[]> mutexes;
        OMP_ONLY(mutexes.reset(new std::mutex[centers.size()]);)
        for(;;) {
            if(centers_cache.size()) {
                for(size_t i = 0; i < centers.size(); ++i)
                    dist::detail::set_cache(centers[i], centers_cache[i], app.get_measure());
            }
            for(auto &c: centers_cpy) c = static_cast<FT>(0);
            OMP_PFOR
            for(size_t i = 0; i < nr; ++i) {
                auto row = blaze::row(assignments, i BLAZE_CHECK_DEBUG);
                for(unsigned j = 0; j < nc; ++j) {
                    row[j] = app(i, centers[j], getcache(j));
                }
                if constexpr(asn_method == SOFT_HARMONIC_MEAN) {
                    row = 1. / row;
                } else {
                    auto mv = blaze::min(row);
                    row = blaze::exp(-row + mv) - mv;
                }
                row *= 1. / blaze::sum(row);
                // And then compute its contribution to the mean of the points.
                // Use stable running mean calculation
            }
            bool restart = false;
            for(size_t i = 0; i < centers.size(); ++i) {
                if(blaze::sum(blaze::column(assignments, i)) == 0.)
                    throw TODOError("TODO: reassignment for support goes to 0");
            }
            if(restart) continue;
            // Now points have been assigned, and we now perform center assignment
            CentroidPolicy::perform_soft_assignment(
                assignments, app.row_sums(),
                OMP_ONLY(mutexes.get(),)
                app.data(), centers_cpy, weight_cv.get(), app.get_measure()
            );
        }
        get_center_change_distance();
        if(ret != UNFINISHED) goto end;
        throw NotImplementedError("Not yet finished");
    }
    end: {
        if(centers_cache.size()) {
            for(size_t i = 0; i < centers.size(); ++i)
                dist::detail::set_cache(centers[i], centers_cache[i], app.get_measure());
        }
        if constexpr(asn_method == HARD) {
            OMP_PFOR
            for(size_t i = 0; i < app.size(); ++i) {
                const auto asn = assignments[i];
                retcost[i] = app(i, centers[asn], getcache(asn));
            }
        } else {
            OMP_PFOR
            for(size_t i = 0; i < app.size(); ++i) {
                for(size_t j = 0; j < centers.size(); ++j) {
                    retcost(i, j) = app(i, centers[j], getcache(j));
                }
            }
        }
    }
    return ret;
}



template<typename FT, typename IT, Assignment asn_method=HARD, CenterOrigination co=INTRINSIC>
void update_defaults_with_measure(ClusteringTraits<FT, IT, asn_method, co> &ct, dist::DissimilarityMeasure measure) {
    if(ct.opt == DEFAULT_OPT) {
        switch(measure) {
            case dist::L2:
            case dist::SQRL2:
            case dist::L1: case dist::TVD:
            case dist::COSINE_DISTANCE:
            case dist::PROBABILITY_COSINE_DISTANCE:
            case dist::LLR: case dist::UWLLR:
            case dist::HELLINGER: case dist::BHATTACHARYYA_DISTANCE: case dist::BHATTACHARYYA_METRIC:
                ct.opt = EXPECTATION_MAXIMIZATION; break;
            /*
             * Bregman Divergences, LLR, cosine distance use the (weighted) mean of each
             * point, in either soft or hard clustering.
             * TVD and L1 use the feature-wise median.
             * Scores are either calculated with softmax distance or harmonic softmax
             */
            case dist::ORACLE_METRIC: case dist::ORACLE_PSEUDOMETRIC: case dist::WASSERSTEIN:
                /* otherwise, use metric kmedian */
                ct.opt = METRIC_KMEDIAN; break;
            default:
                if(dist::detail::is_bregman(measure)) {
                    ct.opt = EXPECTATION_MAXIMIZATION;
                    break;
                }
        }
    }
    if(ct.approx == DEFAULT_APPROX) {
        if(ct.opt == EXPECTATION_MAXIMIZATION) ct.approx = BICRITERIA;
        else ct.approx = CONSTANT_FACTOR;
    }
    if(ct.sampling == DEFAULT_SAMPLING) {
        ct.sampling = ct.opt == EXPECTATION_MAXIMIZATION
            ? D2_SAMPLING: THORUP_SAMPLING;
    }
}

template<Assignment asn_method=HARD, CenterOrigination co=INTRINSIC, typename MatrixType, typename IT=uint32_t>
auto perform_clustering(const jsd::DissimilarityApplicator<MatrixType> &app, size_t npoints, unsigned k,
                        const ElementType_t<MatrixType> *weights=nullptr,
                        CenterSamplingType csample=DEFAULT_SAMPLING,
                        OptimizationMethod opt=DEFAULT_OPT,
                        ApproximateSolutionType approx=DEFAULT_APPROX,
                        uint64_t seed=0,
                        size_t max_iter=100, double eps=1e-4)
{
    MINOCORE_REQUIRE(npoints == app.size(), "assumption");
    using FT = typename MatrixType::ElementType;

    // Setup clustering traits
    ClusteringTraits<FT, IT, asn_method, co> clustering_traits;
    clustering_traits.weights = weights;
    clustering_traits.sampling = csample;
    auto &ct = clustering_traits;
    ct.k = k;
    ct.approx = approx;
    ct.seed = seed;
    ct.max_jv_rounds = ct.max_lloyd_iter = max_iter;
    ct.eps = eps;
    auto measure = app.get_measure();
    update_defaults_with_measure(ct, measure);


    // Setup data and helpers
    typename ClusteringTraits<FT, IT, asn_method, co>::centers_t centers;
    typename ClusteringTraits<FT, IT, asn_method, co>::assignments_t assignments;
    typename ClusteringTraits<FT, IT, asn_method, co>::costs_t costs;
    if constexpr(asn_method == HARD) {
        assignments.resize(app.size());
    } else {
        assignments.resize(app.size(), k);
    }


    auto set_metric_return_values = [&](const auto &ret) {
        auto &[cc, asn, retcosts] = ret;
        centers.resize(cc.size());
        if constexpr(co == EXTRINSIC) {
            OMP_PFOR
            for(size_t i = 0; i < cc.size(); ++i) {
                centers[i] = row(app.data(), cc[i], blaze::unchecked);
            }
        } else {
            std::copy(cc.begin(), cc.end(), centers.begin());
        }
        if constexpr(asn_method == HARD) {
            assignments.resize(asn.size());
            std::copy(asn.begin(), asn.end(), assignments.begin());
            costs.resize(retcosts.size());
            std::copy(retcosts.begin(), retcosts.end(), costs.begin());
        } else {
            throw NotImplementedError("Not supported: soft extrinsic clustering");
        }
    };

    // Delegate to solvers and set-up return values
    if(dist::detail::satisfies_d2(measure) || measure == dist::L1 || measure == dist::TOTAL_VARIATION_DISTANCE || co == EXTRINSIC) {
        auto [initcenters, initasn, initcosts] = jsd::make_kmeanspp(app, ct.k, ct.seed, clustering_traits.weights);
        centers.reserve(k);
        for(const auto id: initcenters) {
            centers.emplace_back(row(app.data(), id));
        }
        //std::copy(initasn.begin(), initasn.end(), std::back_inserter(assignments));
        if(co == INTRINSIC || opt == METRIC_KMEDIAN) {
            // Do graph metric calculation
            MINOCORE_REQUIRE(asn_method == HARD, "Can't do soft metric k-median");
            auto metric_ret = perform_cluster_metric_kmedian<IT, FT>(detail::make_aa(app), app.size(), clustering_traits);
            set_metric_return_values(metric_ret);
        } else {
            // Do Lloyd's loop (``kmeans'' algorithm)
            auto ret = perform_lloyd_loop<asn_method>(centers, assignments, app, k, costs, ct.seed, clustering_traits.weights, max_iter, eps);
            if(ret != FINISHED) std::fprintf(stderr, "lloyd loop ret: %s\n", ret == REACHED_MAX_ROUNDS ? "max rounds": "unfinished");
        }
    } else if(dist::detail::satisfies_metric(measure) || dist::detail::satisfies_rho_metric(measure)) {
        MINOCORE_REQUIRE(asn_method == HARD, "Can't do soft metric k-median");
        auto metric_ret = perform_cluster_metric_kmedian<IT, FT>(detail::make_aa(app), app.size(), clustering_traits);
        set_metric_return_values(metric_ret);
    } else {
        throw NotImplementedError("Unsupported: asymmetric measures not supporting D2 sampling");
    }
    return std::make_tuple(std::move(centers), std::move(assignments), std::move(costs));
} // perform_clustering

// Make # points optional
template<Assignment asn_method=HARD, CenterOrigination co=INTRINSIC, typename MatrixType, typename IT=uint32_t>
auto perform_clustering(const jsd::DissimilarityApplicator<MatrixType> &app, unsigned k,
                        const ElementType_t<MatrixType> *weights=nullptr,
                        CenterSamplingType csample=DEFAULT_SAMPLING,
                        OptimizationMethod opt=DEFAULT_OPT,
                        ApproximateSolutionType approx=DEFAULT_APPROX,
                        uint64_t seed=0,
                        size_t max_iter=100, double eps=1e-4)
{
    return perform_clustering<asn_method, co, MatrixType, IT>(app, app.size(), k, weights, csample, opt, approx, seed, max_iter, eps);
}

template<typename FT=float, typename IT=uint32_t, typename OracleType>
auto perform_clustering(const OracleType &app, size_t npoints, unsigned k,
                        const FT *weights=nullptr,
                        CenterSamplingType csample=DEFAULT_SAMPLING,
                        OptimizationMethod opt=DEFAULT_OPT,
                        ApproximateSolutionType approx=DEFAULT_APPROX,
                        uint64_t seed=0,
                        size_t max_iter=100, double eps=ClusteringTraits<FT, IT, HARD, EXTRINSIC>::DEFAULT_EPS)
{
    if(opt == DEFAULT_OPT) opt = METRIC_KMEDIAN;
    MINOCORE_REQUIRE(opt == METRIC_KMEDIAN, "No other method supported for metric clustering");
    if(approx == DEFAULT_APPROX)
        approx = CONSTANT_FACTOR;
    if(csample == DEFAULT_SAMPLING) {
        csample = THORUP_SAMPLING;
    }
    ClusteringTraits<FT, IT, HARD, EXTRINSIC> clustering_traits = make_clustering_traits(npoints, k,
        csample, opt, approx, weights, seed, max_iter, eps);
    typename ClusteringTraits<FT, IT, HARD, INTRINSIC>::centers_t centers;
    typename ClusteringTraits<FT, IT, HARD, INTRINSIC>::assignments_t assignments;
    typename ClusteringTraits<FT, IT, HARD, INTRINSIC>::costs_t costs;
    auto [cc, asn, retcosts] = perform_cluster_metric_kmedian<IT, FT>(app, npoints, clustering_traits);
    centers.resize(cc.size());
    std::copy(cc.begin(), cc.end(), centers.begin());
    assignments.resize(asn.size());
    std::copy(asn.begin(), asn.end(), assignments.begin());
    costs.resize(retcosts.size());
    std::copy(retcosts.begin(), retcosts.end(), costs.begin());
    return std::make_tuple(std::move(centers), std::move(assignments), std::move(costs));
}


} // namespace clustering

} // namespace minocore

#endif /* FGC_CLUSTERING_DISPATCH_H__ */
