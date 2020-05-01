#ifndef FGC_CLUSTERING_H__
#define FGC_CLUSTERING_H__
#include "minocore/dist.h"
#include "minocore/optim/kmedian.h"
#include "minocore/util/exception.h"
#include "minocore/wip/clustering_traits.h"
#include <cstdint>

namespace minocore {

namespace clustering {

using blz::DissimilarityMeasure;
using blz::ElementType_t;

struct CentroidPolicy {
    template<typename VT, bool TF, typename Range, typename VT2=VT, typename RowSums>
    static void perform_average(blz::DenseVector<VT, TF> &ret, const Range &r, const RowSums &rs,
                                const VT2 *wc = static_cast<VT2 *>(nullptr),
                                DissimilarityMeasure measure=static_cast<DissimilarityMeasure>(-1))
    {
        using FT = blz::ElementType_t<VT>;
        if(measure==static_cast<DissimilarityMeasure>(-1)) {
            std::fprintf(stderr, "Die\n");
            std::exit(1);
        }
        if(measure == blz::TOTAL_VARIATION_DISTANCE) {
            if(wc)
                coresets::l1_median(r, ret, wc->data());
            else
                coresets::l1_median(r, ret);
        }
        else if(measure == blz::L1) {
            std::conditional_t<blz::IsSparseMatrix_v<Range>,
                               blz::CompressedMatrix<FT, blz::StorageOrder_v<Range> >,
                               blz::DynamicMatrix<FT, blz::StorageOrder_v<Range> >
            > cm = r * blz::expand(rs, r.columns());
            if(wc)
                coresets::l1_median(cm, ret, wc->data());
            else
                coresets::l1_median(cm, ret);
        } else if(measure == blz::LLR || measure == blz::UWLLR || measure == blz::OLLR) {
            FT total_sum_inv;
            if(wc) {
                total_sum_inv = 1. / blz::dot(rs, *wc);
                ~ret = blaze::sum<blz::columnwise>(r % blz::expand(*wc * rs, r.columns())) * total_sum_inv;
            } else {
                total_sum_inv = 1. / blaze::sum(rs);
                ~ret = blaze::sum<blz::columnwise>(r % blz::expand(rs, r.columns())) * total_sum_inv;
            }
        } else if(wc) {
            assert((~(*wc)).size() == r.rows());
            assert(blz::expand(~(*wc), r.columns()).rows() == r.rows());
            assert(blz::expand(~(*wc), r.columns()).columns() == r.columns());
            auto wsuminv = 1. / blaze::sum(*wc);
            if(!blz::detail::is_probability(measure)) { // e.g., take mean of unscaled values
                ~ret = blaze::sum<blz::columnwise>(r % blz::expand(~(*wc) * rs, r.columns())) * wsuminv;
            } else {                                    // Else take mean of scaled values
                ~ret = blaze::sum<blz::columnwise>(r % blz::expand(~(*wc), r.columns())) * wsuminv;
            }
        } else {
            if(!blz::detail::is_probability(measure)) {
                auto wsuminv = 1. / blaze::sum(rs);
                ~ret = blaze::sum<blz::columnwise>(r % blz::expand(rs, r.columns())) * wsuminv;
            } else {
                ~ret = blz::mean<blz::columnwise>(r);
            }
        }
    }
    template<typename FT, typename Row, typename Src>
    static void do_inc(FT neww, FT cw, Row &ret, const Src &dat, FT row_sum, DissimilarityMeasure measure)
    {
        if(measure == blz::L1 || measure == blz::TOTAL_VARIATION_DISTANCE)
            throw std::invalid_argument("do_inc is only for linearly-calculated means, not l1 median");
        if(cw == 0.) {
            if(blz::detail::is_probability(measure))
                ret = dat;
            else
                ret = dat * row_sum;
        } else {
            auto div = neww / (neww + cw);
            if(blz::detail::is_probability(measure)) {
                ret += (dat - ret) * div;
            } else if(measure == blz::LLR || measure == blz::UWLLR) {
                ret += (dat * row_sum) * neww;
                // Add up total sum and subtract later
                // since there are three weighting factors here:
                // First, partial assignment
                // Then point-wise weights (both of which are in neww)
                // Then, for LLR/UWLLR, there's weighting by the row-sums
            } else {
                // Maintain running mean for full vector value
                ret += (dat * row_sum - ret) * div;
            }
        }
    }

    template<typename VT, bool TF, typename RowSums, typename MatType, typename CenterCon, typename VT2=blz::DynamicVector<blz::ElementType_t<VT>> >
    static void perform_soft_assignment(const blz::DenseMatrix<VT, TF> &assignments, 
        const RowSums &rs,
        OMP_ONLY(std::mutex *mutptr,)
        const MatType &data, CenterCon &newcon,
        const VT2 *wc = static_cast<const VT2 *>(nullptr),
        DissimilarityMeasure measure=static_cast<DissimilarityMeasure>(-1))
    {
        using FT = blz::ElementType_t<VT>;
        if(measure==static_cast<DissimilarityMeasure>(-1)) {
            std::fprintf(stderr, "Die\n");
            std::exit(1);
        }
        if(measure == blz::L1 || measure == blz::TOTAL_VARIATION_DISTANCE) {
            OMP_PFOR
            for(unsigned j = 0; j < newcon.size(); ++j) {
                blz::DV<FT, blz::rowVector> newweights = trans(column(assignments, j));
                if(wc) {
                    newweights *= *wc;
                }
                if(measure == blz::L1) {
                    std::conditional_t<blz::IsDenseMatrix_v<VT>,
                                       blz::DM<FT>, blz::SM<FT>> scaled_data = data % blz::expand(rs, data.columns());
                    coresets::l1_median(scaled_data, newcon[j], newweights.data());
                } else { // TVD
                    coresets::l1_median(data, newcon[j], newweights.data());
                }
            }
        } else {
            blz::DV<FT> summed_contribs(newcon.size(), 0.);
            OMP_PFOR
            for(size_t i = 0; i < data.rows(); ++i) {
                auto item_weight = wc ? wc->operator[](i): static_cast<FT>(1.);
                const auto row_sum = rs[i];
                auto asn(row(assignments, i, blz::unchecked));
                for(size_t j = 0; j < newcon.size(); ++j) {
                    auto &cw = summed_contribs[j];
                    if(auto asnw = asn[j]; asnw > 0.) {
                        auto neww = item_weight * asnw;
                        OMP_ONLY(if(mutptr) mutptr->lock();)
                        do_inc(neww, cw, newcon[j], row(data, i, blz::unchecked), row_sum, measure);
                        OMP_ONLY(if(mutptr) mutptr->unlock();)
                        OMP_ATOMIC
                        cw += neww;
                    }
                }
            }
            if(measure == blz::LLR || measure == blz::UWLLR || measure == blz::OLLR) {
                OMP_PFOR
                for(auto i = 0u; i < newcon.size(); ++i)
                    newcon[i] *= 1. / blz::dot(column(assignments, i), rs);
            }
        }
    }
};

template<typename IT=uint32_t, typename MatrixType, typename WFT=blz::ElementType_t<MatrixType> >
auto perform_cluster_metric_kmedian(const jsd::DissimilarityApplicator<MatrixType> &app, unsigned k, uint64_t seed=0, const WFT *weights=static_cast<WFT *>(nullptr))
{
    double v = 0.;
    for(size_t i = 0; i < k; ++i) {
        v += app(i, i) * weights[i] - seed;
    }
    std::fprintf(stderr, "Useless computation to avoid warnings\n");
    blz::DV<IT> cc(k), asn(app.size());
    blz::DV<blz::ElementType_t<MatrixType>> retcosts;
    throw NotImplementedError();
    return std::make_tuple(cc, asn, retcosts);
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
    if(blz::detail::needs_logs(app.measure_) || blz::detail::needs_sqrt(app.measure_))
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
                return value + blaze::sum(blz::abs(center - centers[ind]));
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
    using cv_t = blaze::CustomVector<WFT, blz::unaligned, blz::unpadded, blz::rowVector>;
    std::unique_ptr<cv_t> weight_cv;
    if(weights) {
        weight_cv.reset(new cv_t(const_cast<WFT *>(weights), app.size()));
    }
    auto getcache = [&] (size_t j) {
        return centers_cache.size() ? &centers_cache[j]: static_cast<decltype(&centers_cache[j])>(nullptr);
    };
    std::fprintf(stderr, "TODO: modify this to select new centers if any loses all "
                         "support. This would use the variable k (%u)"
                         ", which is currently unused otherwise.\n",
                unsigned(k));
    if constexpr(asn_method == HARD) {
        std::vector<std::vector<uint32_t>> assigned(centers.size());
        OMP_ONLY(std::unique_ptr<std::mutex[]> mutexes(new std::mutex[centers.size()]);)
        for(;;) {
            // Do it forever
            if(centers_cache.size()) {
                for(size_t i = 0; i < centers.size(); ++i)
                    blz::detail::set_cache(centers[i], centers_cache[i], app.measure_);
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
            bool restart = false;
            for(size_t i = 0; i < assignments.size(); ++i) {
                if(assigned[i].empty()) {
                    uint64_t idx = rng() % app.size();
                    centers[i] = row(app.data(), idx BLAZE_CHECK_DEBUG);
                    restart = true;
                }
            }
            if(restart) continue;
            // Make assignments
            for(size_t i = 0; i < centers_cpy.size(); ++i) {
                auto &cref = centers_cpy[i];
                auto &assigned_ids = assigned[i];
                shared::sort(assigned_ids.begin(), assigned_ids.end()); // Better access pattern
                if(weight_cv) {
                    auto wsel = blz::elements(*weight_cv, assigned_ids.data(), assigned_ids.size());
                    CentroidPolicy::perform_average(
                        cref,
                        rows(mat, assigned_ids.data(), assigned_ids.size()),
                        blz::elements(app.row_sums(), assigned_ids.data(), assigned_ids.size()),
                        &wsel, app.measure_
                    );
                } else {
                    using ptr_t = std::add_pointer_t<decltype(blz::elements(*weight_cv, assigned_ids.data(), assigned_ids.size()))>;
                    CentroidPolicy::perform_average(
                        cref,
                        rows(mat, assigned_ids.data(), assigned_ids.size()),
                        blz::elements(app.row_sums(), assigned_ids.data(), assigned_ids.size()),
                        static_cast<ptr_t>(nullptr), app.measure_
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
                    blz::detail::set_cache(centers[i], centers_cache[i], app.measure_);
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
                    auto mv = blz::min(row);
                    row = blz::exp(-row + mv) - mv;
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
                app.data(), centers_cpy, weight_cv.get(), app.measure_
            );
        }
        get_center_change_distance();
        if(ret != UNFINISHED) goto end;
        throw NotImplementedError("Not yet finished");
    }
    end: {
        if(centers_cache.size()) {
            for(size_t i = 0; i < centers.size(); ++i)
                blz::detail::set_cache(centers[i], centers_cache[i], app.measure_);
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


template<Assignment asn_method=HARD, CenterOrigination co=INTRINSIC, typename MatrixType, typename IT=uint32_t>
auto perform_clustering(const jsd::DissimilarityApplicator<MatrixType> &app, unsigned k, CenterSamplingType csample=DEFAULT_SAMPLING,
                        const blz::ElementType_t<MatrixType> *weights=nullptr, uint64_t seed=0, OptimizationMethod opt=DEFAULT_OPT,
                        size_t max_iter=100, double eps=1e-4)
{
    using FT = typename MatrixType::ElementType;
    ClusteringTraits<FT, IT, asn_method, co> clustering_traits;
    clustering_traits.sampling = csample;
    typename ClusteringTraits<FT, IT, asn_method, co>::centers_t centers;
    typename ClusteringTraits<FT, IT, asn_method, co>::assignments_t assignments;
    typename ClusteringTraits<FT, IT, asn_method, co>::costs_t costs;
    if constexpr(asn_method == HARD) {
        assignments.resize(app.size());
    } else {
        assignments.resize(app.size(), k);
    }
    auto measure = app.measure_;
    if(opt == DEFAULT_OPT) {
        switch(measure) {
            case blz::L2:
            case blz::SQRL2:
            case blz::L1: case blz::TVD:
            case blz::COSINE_DISTANCE:
            case blz::PROBABILITY_COSINE_DISTANCE:
            case blz::LLR: case blz::UWLLR:
            case blz::HELLINGER: case blz::BHATTACHARYYA_DISTANCE:
                opt = EXPECTATION_MAXIMIZATION; break;
            /*
             * Bregman Divergences, LLR, cosine distance use the (weighted) mean of each
             * point, in either soft or hard clustering.
             * TVD and L1 use the feature-wise median.
             * Scores are either calculated with softmax distance or harmonic softmax
             */
            case blz::ORACLE_METRIC: case blz::ORACLE_PSEUDOMETRIC: case blz::BHATTACHARYYA_METRIC: case blz::WASSERSTEIN:
                /* otherwise, use metric kmedian */
                opt = METRIC_KMEDIAN; break;
            default:
                if(blz::detail::is_bregman(app.measure_)) {
                    opt = EXPECTATION_MAXIMIZATION;
                    break;
                }
        }
    }

    auto set_metric_return_values = [&](auto &ret) {
        auto &[cc, asn, retcosts] = ret;
        centers.reserve(cc.size());
        if constexpr(co == EXTRINSIC) {
            for(size_t i = 0; i < cc.size(); ++i) {
                centers.emplace_back(row(app.data(), cc[i], blz::unchecked));
            }
        } else {
            centers.resize(cc.size());
            std::copy(cc.begin(), cc.end(), centers.begin());
        }
        if constexpr(asn_method == HARD) {
            assignments.resize(asn.size());
            std::copy(asn.begin(), asn.end(), assignments.begin());
            costs.resize(retcosts.size());
            std::copy(retcosts.begin(), retcosts.end(), costs.begin());
        } else {
            throw NotImplementedError("Not supported: soft extrinsinc clustering");
        }
    };
    if(blz::detail::satisfies_d2(measure) || measure == blz::L1 || measure == blz::TOTAL_VARIATION_DISTANCE || co == EXTRINSIC) {
        auto [initcenters, initasn, initcosts] = jsd::make_kmeanspp(app, k, seed, weights);
        centers.reserve(k);
        for(const auto id: initcenters) {
            centers.emplace_back(row(app.data(), id));
        }
        //std::copy(initasn.begin(), initasn.end(), std::back_inserter(assignments));
        if(co == INTRINSIC || opt == METRIC_KMEDIAN) {
            // Do graph metric calculation
            MINOCORE_REQUIRE(asn_method == HARD, "Can't do soft metric k-median");
            auto metric_ret = perform_cluster_metric_kmedian<IT>(app, k, seed, weights);
            set_metric_return_values(metric_ret);
        } else {
            // Do Lloyd's loop (``kmeans'' algorithm)
            auto ret = perform_lloyd_loop<asn_method>(centers, assignments, app, k, costs, seed, weights, max_iter, eps);
            if(ret != FINISHED) std::fprintf(stderr, "lloyd loop ret: %s\n", ret == REACHED_MAX_ROUNDS ? "max rounds": "unfinished");
        }
    } else if(blz::detail::is_symmetric(measure)) {
        throw std::runtime_error("Not implemented: symmetric measure clustering. This method should perform sampling (governed by the csample variable)"
                                  ", followed by facility location, and finished by local search.");
        MINOCORE_REQUIRE(asn_method == HARD, "Can't do soft metric k-median");
        auto metric_ret = perform_cluster_metric_kmedian<IT>(app, k, seed, weights);
        set_metric_return_values(metric_ret);
    } else {
        throw NotImplementedError("Unsupported: asymmetric measures not supporting D2 sampling");
    }
    return std::make_tuple(std::move(centers), std::move(assignments), std::move(costs));
} // perform_clustering


} // namespace clustering

} // namespace minocore

#endif /* FGC_CLUSTERING_H__ */
