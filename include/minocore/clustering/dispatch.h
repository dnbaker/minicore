#ifndef FGC_CLUSTERING_DISPATCH_H__
#define FGC_CLUSTERING_DISPATCH_H__
#include "minocore/dist.h"
#include "minocore/optim/jv_solver.h"
#include "minocore/optim/lsearch.h"
#include "minocore/optim/oracle_thorup.h"
#include "minocore/util/exception.h"
#include "minocore/clustering/traits.h"
#include "minocore/clustering/sampling.h"
#include "minocore/clustering/centroid.h"

#include "boost/iterator/zip_iterator.hpp"
#include "diskmat/diskmat.h"

namespace minocore {

namespace clustering {

using dist::DissimilarityMeasure;
using blaze::ElementType_t;
using diskmat::PolymorphicMat;
using boost::make_zip_iterator;

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
    if constexpr(asn_method == HARD) {
        if(retcost.size() != app.size()) retcost.resize(app.size());
    } else {
        // asn_method == SOFT || asn_method == SOFT_HARMONIC_MEAN
        retcost.resize(app.size(), k);
    }
    assert(retcost.size() == app.size() || !std::fprintf(stderr, "retcost size: %zu. app size: %zu\n", retcost.size(), app.size()));
    if(co != EXTRINSIC) throw std::invalid_argument("Must be extrinsic for Lloyd's");
    using FT = ElementType_t<MatrixType>;
    auto &mat = app.data();
    const size_t npoints = app.size();
    CentersType centers_cpy(centers), centers_cache;
    MINOCORE_REQUIRE(centers.size() == k, "Must have the correct number of centers");
    const auto measure = app.get_measure();
    if(dist::detail::needs_logs(measure) || dist::detail::needs_sqrt(measure))
        centers_cache.resize(k);
    FT current_cost = std::numeric_limits<FT>::max(), first_cost = current_cost;
    //PRETTY_SAY << "Beginning\n";
    LloydLoopResult ret = UNFINISHED;
    wy::WyRand<uint64_t> rng(seed);
    size_t iternum = 0;
    // HEY DANIEL WHEN YOU GET BACK HERE
    // You are removing the center_distance
    // and instead calculating the objective function
    // and terminating when the change in objective function is less than eps * first cost.
    using cv_t = blaze::CustomVector<WFT, blaze::unaligned, blaze::unpadded, blaze::rowVector>;
    std::unique_ptr<cv_t> weight_cv;
    if(weights) {
        weight_cv.reset(new cv_t(const_cast<WFT *>(weights), npoints));
    }
    auto getcache = [&] (size_t j) {
        decltype(&centers_cache[j]) ret = nullptr;
        if(centers_cache.size()) ret = &centers_cache[j];
        return ret;
    };
    assert(centers_cache.empty() || getcache(0) == nullptr);
    auto getcost = [&]() {
        if constexpr(asn_method == HARD) {
            return weight_cv ? blz::sum(retcost * *weight_cv): blz::sum(retcost);
        } else {
#ifndef NDEBUG
            // Ensure that the assignments are as expected.
            for(size_t i = 0; i < assignments.rows(); ++i) {
                auto r(row(assignments, i));
                auto cr(row(retcost, i));
                auto maxi = std::max_element(r.begin(), r.end()) - r.begin();
                auto mini = std::min_element(cr.begin(), cr.end()) - cr.begin();
                //std::cerr << "mini: " << mini << '\n';
                //std::cerr << "maxi: " << maxi << '\n';
                assert(std::abs(blaze::sum(r) - 1.) < 1e-4);
                assert(maxi == mini || r[maxi] == r[mini] || cr[mini] == cr[maxi]
                      || &(std::cerr << r << '\n' << cr << '\n') == nullptr);
            }
#endif
            if(weight_cv) {
                auto ew = blaze::expand(*weight_cv, app.data().columns());
                std::fprintf(stderr, "expanded weight shape: %zu/%zu. asn: %zu/%zu\n", ew.rows(), ew.columns(), assignments.rows(), assignments.columns());
                return blaze::sum(assignments % retcost % ew);
            } else {
                return blaze::sum(assignments % retcost);
            }
        }
    };
    auto check = [&]() {
        ++iternum;
        if(first_cost == std::numeric_limits<FT>::max()) first_cost = getcost();
        else {
            FT itercost = getcost();
            if(current_cost == std::numeric_limits<FT>::max()) {
                current_cost = itercost;
                assert(current_cost != std::numeric_limits<FT>::max());
            } else {
                if(std::abs(itercost - current_cost) < eps * first_cost) { // consider taking sign into account here
                    PRETTY_SAY << "Itercost: " << itercost << " vs current " << current_cost << " with diff " << std::abs(itercost - current_cost)
                               << "compared to first cost of " << first_cost << " with eps = " << eps << ".\n";
                    return FINISHED;
                }
                if(iternum == max_iter)
                    return REACHED_MAX_ROUNDS;
            }
            current_cost = itercost;
        }
        PRETTY_SAY << "iternum: " << iternum << '\n';
        return UNFINISHED;
    };
    auto soft_assignments = [&]() {
        if constexpr(asn_method != HARD) {
            OMP_PFOR
            for(size_t i = 0; i < npoints; ++i) {
                auto row = blaze::row(retcost, i BLAZE_CHECK_DEBUG);
                for(unsigned j = 0; j < centers.size(); ++j) {
                    row[j] = app(i, centers[j], getcache(j), measure);
                }
                auto asnrow = blaze::row(assignments, i BLAZE_CHECK_DEBUG);
                if constexpr(asn_method == SOFT_HARMONIC_MEAN) {
                    asnrow = 1. / row;
                } else {
                    auto mv = blaze::min(row);
                    assert(mv >= 0.);
                    asnrow = blaze::exp(-row + mv);
                    assert(blaze::min(asnrow) >= 0.);
                }
                asnrow *= 1. / blaze::sum(asnrow);
                assert(blaze::min(asnrow) >= 0.);
                PRETTY_SAY << "row " << row << " yields " << asnrow << " with max " << blz::max(asnrow) << ", min " << blz::min(asnrow) <<'\n';
            }
        }
    };
    if constexpr(asn_method == HARD) {
        std::vector<std::vector<uint32_t>> assigned(k);
        OMP_ONLY(std::unique_ptr<std::mutex[]> mutexes(new std::mutex[k]);)
        for(;;) {
            // Do it forever
            if(centers_cache.size()) {
                PRETTY_SAY << "Setting centers cache for measure " << dist::detail::prob2str(measure) << '\n';
                for(unsigned i = 0; i < k; ++i)
                    dist::detail::set_cache(centers[i], centers_cache[i], measure);
            }
            for(auto &i: assigned) i.clear();
            OMP_PFOR
            for(size_t i = 0; i < npoints; ++i) {
                auto dist = app(i, centers[0], getcache(0), measure);
                unsigned asn = 0;
                for(unsigned j = 1; j < k; ++j) {
                    auto newdist = app(i, centers[j], getcache(j), measure);
                    if(newdist < dist) {
                        asn = j;
                        dist = newdist;
                    }
                }
                retcost[i] = dist;
                assignments[i] = asn;
                {
                    OMP_ONLY(std::unique_lock<std::mutex> lock(mutexes[asn]);)
                    assigned[asn].push_back(i);
                }
            }
            // Check termination condition
            if(auto rc = check(); rc != UNFINISHED) {
                ret = rc;
                goto end;
            }
            blaze::SmallArray<uint32_t, 16> centers_to_restart;
            for(unsigned i = 0; i < k; ++i)
                if(assigned[i].empty())
                    centers_to_restart.pushBack(i);
            if(auto restartn = centers_to_restart.size()) {
                // Use D^2 sampling to stayrt a new cluster
                // And then restart the loop
                assert(retcost.size() == npoints);
                retcost = std::numeric_limits<FT>::max();
                OMP_PFOR
                for(size_t i = 0; i < npoints; ++i) {
                    for(size_t j = 0; j < k; ++j) {
                        if(assigned[j].empty()) continue;
                        auto fc = app(i, centers[j], getcache(j), measure);
                        if(fc < retcost[i]) retcost[i] = fc;
                    }
                }
                blaze::DynamicVector<FT> csum(npoints);
                std::uniform_real_distribution<FT> urd;
                for(size_t i = 0; i < restartn;) {
                    std::partial_sum(retcost.data(), retcost.data() + retcost.size(), csum.data());
                    auto newp = std::lower_bound(csum.data(), csum.data() + csum.size(), urd(rng) * csum[csum.size() - 1])
                                - csum.data();
                    centers[centers_to_restart[i]] = row(app.data(), newp, blaze::unchecked);
                    if(++i != restartn) {
                        OMP_PFOR
                        for(size_t i = 0; i < npoints; ++i)
                            retcost[i] = std::min(retcost[i], app(i, newp));
                    }
                }
                continue; // Reassign, re-center, and re-compute
            }
            // Make centers
            for(size_t i = 0; i < centers_cpy.size(); ++i) {
                auto &cref = centers_cpy[i];
                auto &assigned_ids = assigned[i];
                shared::sort(assigned_ids.begin(), assigned_ids.end()); // Better access pattern
                auto aidptr = assigned_ids.data();
                const size_t nid = assigned_ids.size();
                auto rowsel = rows(mat, aidptr, nid);
                auto sumsel = blaze::elements(app.row_sums(), aidptr, nid);
                if(weight_cv) {
                    auto wsel = blaze::elements(*weight_cv, aidptr, nid);
                    CentroidPolicy::perform_average(cref, rowsel, sumsel, &wsel, measure);
                } else {
                    CentroidPolicy::perform_average(cref, rowsel, sumsel,
                        static_cast<decltype(blaze::elements(*weight_cv, aidptr, nid)) *>(nullptr), measure
                    );
                    //PRETTY_SAY << "Center " << i << " is " << cref << '\n';
                    PRETTY_SAY << "Difference between previous center and new center is " << blz::sqrL2Dist(cref, centers[i]) << '\n';
                }
            }
            // Set the returned values to be the last iteration's.
            centers = centers_cpy;
        }
    } else {
        if(assignments.rows() != npoints || assignments.columns() != centers.size()) {
            assignments.resize(npoints, centers.size());
        }
        std::unique_ptr<std::mutex[]> mutexes;
        OMP_ONLY(mutexes.reset(new std::mutex[centers.size()]);)
        for(;;) {
            if(centers_cache.size()) {
                for(size_t i = 0; i < centers.size(); ++i)
                    dist::detail::set_cache(centers[i], centers_cache[i], measure);
            }
            for(auto &c: centers_cpy) c = static_cast<FT>(0);
            soft_assignments();
            assert(blz::sum(assignments) - assignments.rows() < 1e-3 * assignments.rows());
            for(size_t i = 0; i < centers.size(); ++i)
                if(blaze::sum(blaze::column(assignments, i)) == 0.)
                    throw TODOError("TODO: reassignment for support goes to 0");
            // Check termination condition
            if(auto rc = check(); rc != UNFINISHED) {
                ret = rc;
                goto end;
            }
            // Now points have been assigned, and we now perform center assignment
            CentroidPolicy::perform_soft_assignment(
                assignments, app.row_sums(),
                OMP_ONLY(mutexes.get(),)
                app.data(), centers_cpy, weight_cv.get(), measure
            );
        }
        std::swap(centers_cpy, centers);
    }
    end: {
        if(centers_cache.size()) {
            for(size_t i = 0; i < centers.size(); ++i)
                dist::detail::set_cache(centers[i], centers_cache[i], measure);
        }
        soft_assignments();
    }
    DBG_ONLY(if(ret == FINISHED) PRETTY_SAY << "Completed Lloyd's loop in " << iternum << " iterations\n";)
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
    auto ct = make_clustering_traits<FT, IT, asn_method, co>(npoints, k,
        csample, opt, approx, weights, seed, max_iter, eps);
    using ct_t = decltype(ct);
    auto measure = app.get_measure();
    update_defaults_with_measure(ct, measure);

    // and helpers
    typename ct_t::centers_t centers;
    centers.reserve(k);
    typename ct_t::assignments_t assignments;
    typename ct_t::costs_t costs;
    if constexpr(asn_method == HARD) {
        assignments.resize(app.size());
    } else {
        assignments.resize(app.size(), k);
    }
    PRETTY_SAY << "Assignments sized.\n";


    auto set_metric_return_values = [&](const auto &ret) {
        MINOCORE_REQUIRE(asn_method == HARD, "Not supported: soft extrinsic clustering");
        auto &[cc, asn, retcosts] = ret;
        centers.resize(cc.size());
        if constexpr(co == EXTRINSIC) {
            OMP_PFOR
            for(size_t i = 0; i < cc.size(); ++i) {
                centers[i] = row(app.data(), cc[i], blaze::unchecked);
            }
        } else std::copy(cc.begin(), cc.end(), centers.begin()); // INTRINSIC
        if constexpr(asn_method == HARD) {
            assignments.resize(asn.size());
            std::copy(asn.begin(), asn.end(), assignments.begin());
            costs.resize(retcosts.size());
            std::copy(retcosts.begin(), retcosts.end(), costs.begin());
        }
    };

    // Delegate to solvers and set-up return values
    if(dist::detail::satisfies_d2(measure) || measure == dist::L1 || measure == dist::TOTAL_VARIATION_DISTANCE || co == EXTRINSIC) {
        auto [initcenters, initasn, initcosts] = jsd::make_kmeanspp(app, ct.k, ct.seed, ct.weights);
        assert(initcenters.size() == k);
        if(co == INTRINSIC || opt == METRIC_KMEDIAN) {
            PRETTY_SAY << "Performing metric clustering\n";
            // Do graph metric calculation
            MINOCORE_REQUIRE(asn_method == HARD, "Can't do soft metric k-median");
            auto metric_ret = perform_cluster_metric_kmedian<IT, FT>(detail::make_aa(app), app.size(), ct);
            set_metric_return_values(metric_ret);
        } else {
            PRETTY_SAY << "Setting centers with D2\n";
            for(const auto id: initcenters)
                centers.emplace_back(row(app.data(), id));
            assert(centers.size() == k);
            PRETTY_SAY << "Beginning lloyd loop\n";
            // Perform EM
            if(auto ret = perform_lloyd_loop<asn_method>(centers, assignments, app, k, costs, ct.seed, ct.weights, max_iter, eps))
                std::fprintf(stderr, "lloyd loop ret: %s\n", ret == REACHED_MAX_ROUNDS ? "max rounds": "unfinished");
        }
    } else if(dist::detail::satisfies_metric(measure) || dist::detail::satisfies_rho_metric(measure)) {
        MINOCORE_REQUIRE(asn_method == HARD, "Can't do soft metric k-median");
        auto metric_ret = perform_cluster_metric_kmedian<IT, FT>(detail::make_aa(app), app.size(), ct);
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
    // Setup
    if(opt == DEFAULT_OPT) opt = METRIC_KMEDIAN;
    if(approx == DEFAULT_APPROX) approx = CONSTANT_FACTOR;
    if(csample == DEFAULT_SAMPLING) csample = THORUP_SAMPLING;
    MINOCORE_REQUIRE(opt == METRIC_KMEDIAN, "No other method supported for metric clustering");
    auto clustering_traits = make_clustering_traits<FT, IT, HARD, EXTRINSIC>(npoints, k,
        csample, opt, approx, weights, seed, max_iter, eps);
    using ct_t = decltype(clustering_traits);

    // Cluster
    auto [cc, asn, retcosts] = perform_cluster_metric_kmedian<IT, FT>(app, npoints, clustering_traits);

    // Return
    typename ct_t::centers_t centers(cc.size());
    typename ct_t::assignments_t assignments(asn.size());
    typename ct_t::costs_t costs(retcosts.size());
    std::copy(cc.begin(), cc.end(), centers.begin());
    std::copy(asn.begin(), asn.end(), assignments.begin());
    std::copy(retcosts.begin(), retcosts.end(), costs.begin());
    return std::make_tuple(std::move(centers), std::move(assignments), std::move(costs));
}


} // namespace clustering

} // namespace minocore

#endif /* FGC_CLUSTERING_DISPATCH_H__ */
