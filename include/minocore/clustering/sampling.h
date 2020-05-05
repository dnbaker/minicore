#ifndef CLUSTERING_SAMPLING_H__
#define CLUSTERING_SAMPLING_H__
#include "minocore/clustering/traits.h"
#include "minocore/optim/oracle_thorup.h"

namespace minocore {

namespace clustering {

template<typename IT, typename FT>
struct MetricSelectionResult: public std::tuple<std::vector<IT>, blz::DV<FT>, std::vector<IT>, blz::DM<FT> > {
	auto &selected() {return std::get<0>(*this);}
	const auto &selected() const {return std::get<0>(*this);}
	auto &costs() {return std::get<1>(*this);}
	const auto &costs() const {return std::get<1>(*this);}
	auto &assignments() {return std::get<2>(*this);}
	const auto &assignments() const {return std::get<2>(*this);}
	auto &facility_cost_matrix() {return std::get<3>(*this);}
	const auto &facility_cost_matrix() const {return std::get<3>(*this);}
};



template<typename OracleType, typename IT, typename FT, Assignment asn=HARD, CenterOrigination co=EXTRINSIC>
MetricSelectionResult<IT, FT>
select_uniform_random(const OracleType &oracle, size_t np, ClusteringTraits<FT, IT, asn, co> opts)
{
    assert(opts.k != (unsigned)-1);
	MetricSelectionResult<IT, FT> ret;
    size_t nsamp = std::min(size_t(std::ceil(opts.k * opts.approx_mul)), np);
    std::vector<IT> selected;
    std::mt19937_64 rng(opts.seed);
    schism::Schismatic<IT> modder(np);
    blz::DV<FT> costs(np, std::numeric_limits<FT>::max());
    std::vector<IT> assignments(np);
    shared::flat_hash_set<IT> sel;
    do {
        IT next;
        do next = modder.mod(rng());
        while(sel.find(next) != sel.end());
        OMP_PFOR
        for(size_t i = 0; i < np; ++i) {
            if(costs[i] == 0.) continue;
            auto c = oracle(next, i);
            if(c < costs[i])
                costs[i] = c, assignments[i] = selected.size();
        }
        sel.insert(next);
        selected.push_back(next);
    } while(selected.size() < nsamp);
    std::get<0>(ret) = std::move(selected);
    std::get<1>(ret) = std::move(costs);
    std::get<2>(ret) = std::move(assignments);
    return ret;
}

template<typename OracleType, typename IT, typename FT, Assignment asn=HARD, CenterOrigination co=EXTRINSIC>
MetricSelectionResult<IT, FT>
select_greedy(const OracleType &oracle, size_t np, ClusteringTraits<FT, IT, asn, co> opts)
{
    assert(opts.k != (unsigned)-1);
	MetricSelectionResult<IT, FT> ret;
    size_t nsamp = std::min(size_t(std::ceil(opts.k * opts.approx_mul)), np);
    blz::DV<FT> costs(np);
    IT next = std::mt19937_64(opts.seed)() % np;
    std::vector<IT> selected{next}, assignments(np, next);
    costs[next] = 0.;
    OMP_PFOR
    for(size_t i = 0; i < np; ++i) {
        if(unlikely(i == next)) continue;
        costs[i] = oracle(i, next);
    }

    while(selected.size() < nsamp) {
        next = std::max_element(costs.data(), costs.data() + costs.size()) - costs.data();
        costs[next] = 0.;
        assignments[next] = next;
        OMP_PFOR
        for(size_t i = 0; i < np; ++i) {
            if(unlikely(i == next) || costs[i] == 0.) continue;
            if(auto newcost = oracle(i, next); newcost < costs[i]) {
                costs[i] = newcost;
                assignments[i] = next;
            }
        }
        selected.push_back(next);
    }
    std::get<0>(ret) = std::move(selected);
    std::get<1>(ret) = std::move(costs);
    std::get<2>(ret) = std::move(assignments);
    return ret;
}

template<typename OracleType, typename IT, typename FT, Assignment asn=HARD, CenterOrigination co=EXTRINSIC>
MetricSelectionResult<IT, FT>
select_d2(const OracleType &oracle, size_t np, ClusteringTraits<FT, IT, asn, co> opts) {
	MetricSelectionResult<IT, FT> ret;
    size_t nsamp = std::min(size_t(std::ceil(opts.k * opts.approx_mul)), np);
    blz::DV<FT> costs(np);
    std::mt19937_64 mt(opts.seed);
    IT next = mt() % np;
    std::vector<IT> selected{next}, assignments(np, next);
    costs[next] = 0.;
    OMP_PFOR
    for(size_t i = 0; i < np; ++i) {
        if(unlikely(i == next)) continue;
        costs[i] = oracle(i, next);
    }
    auto cdf = std::make_unique<FT[]>(np);
    FT *const cdfbeg = cdf.get(), *const cdfend = cdfbeg + np;
    do {
        std::partial_sum(costs.data(), costs.data() + costs.size(), cdfbeg);
        std::uniform_real_distribution<double> dist;
        IT id;
        do {
            id = std::lower_bound(cdfbeg, cdfend, cdfend[-1] * dist(mt)) - cdfbeg;
        } while(std::find(selected.begin(), selected.end(), id) != selected.end());
        selected.push_back(id);
        costs[id] = 0.;
        assignments[id] = id;
        OMP_PFOR
        for(IT i = 0; i < np; ++i) {
            if(costs[i] == 0.) continue;
            if(auto newcost = oracle(i, id); newcost < costs[i]) {
                costs[i] = newcost;
                assignments[i] = id;
            }
        }
    } while(selected.size() < nsamp);
    std::get<0>(ret) = std::move(selected);
    std::get<1>(ret) = std::move(costs);
    std::get<2>(ret) = std::move(assignments);
    return ret;
}

#if 0
template<typename OracleType, typename IT, typename FT, Assignment asn=HARD, CenterOrigination co=EXTRINSIC>
MetricSelectionResult<IT, FT>
select_facilities(const OracleType &app, size_t np, AnalysisOpts opts, const WFT *weights=nullptr, const char *save_distmat_path=nullptr)
{
    auto select_start = std::chrono::high_resolution_clock::now();
	MetricSelectionResult<IT, FT> ret;
    if(opts.compute_full) {
        std::cerr << "Computing full distance\n";
        dm::DistanceMatrix<float, 0, mmap_distmat ? dm::DM_MMAP: dm::DM_DEFAULT> distmat(
            np, 0, std::string(save_distmat_path ? save_distmat_path: ""));
        minocore::util::Timer t("distmat calc");
        for(size_t i = 0; i < np; ++i) {
            auto [ptr, extent] = distmat.row_span(i);
            const auto offset = i + 1;
#ifdef _OPENMP
#           pragma omp parallel for schedule(dynamic)
#endif
            for(size_t j = 0; j < extent; ++j) {
                ptr[j] = app(i, j + offset);
            }
        }
        t.report();
        t.reset();
        switch(opts.algo_) {
            case THORUP_SAMPLING:
                opts.thorup_iter = 1;
                [[fallthrough]];
            case ITERATED_THORUP_SAMPLING:
                std::tie(std::get<0>(ret), std::get<1>(ret), std::get<2>(ret))
                    = minocore::iterated_oracle_thorup_d(distmat, np, opts.k, opts.thorup_iter, opts.thorup_sub_iter, weights, opts.npermult, 3, 0.5, opts.seed);
            break;
            case GREEDY_ADDITION: ret = select_greedy(distmat, np, opts); break;
            case D2_SAMPLING: ret = select_d2(distmat, np, opts);         break;
        }
        auto &centers = std::get<0>(ret);
        minocore::shared::flat_hash_map<uint32_t, uint32_t> select_set;
        select_set.reserve(centers.size());
        for(size_t i = 0; i < centers.size(); ++i)
            select_set[centers[i]] = i;
        auto &retdm = std::get<3>(ret);
        retdm.resize(std::get<0>(ret).size(), np);
        for(size_t i = 0; i < centers.size(); ++i) {
            const auto cid = centers[i];
            auto rowptr = row(retdm, i);
            OMP_PFOR
            for(size_t j = 0; j < np; ++j) {
                if(unlikely(j == cid)) rowptr[j] = 0.;
                rowptr[j] =  distmat(cid, j);
            }
        }
        assert(blz::min(retdm) >= 0.);
    } else {
        auto caching_app = dash::make_row_caching_oracle_wrapper<
            minocore::shared::flat_hash_map, /*is_symmetric=*/ true, /*is_threadsafe=*/true
        >(app, np);
        switch(opts.algo_) {
            case THORUP_SAMPLING:
                opts.thorup_iter = 1;
                [[fallthrough]];
            case ITERATED_THORUP_SAMPLING:
                std::tie(std::get<0>(ret), std::get<1>(ret), std::get<2>(ret))
                    = minocore::iterated_oracle_thorup_d(caching_app, np, opts.k, opts.thorup_iter, opts.thorup_sub_iter, weights, opts.npermult, 3, 0.5, opts.seed);
            break;
            case GREEDY_ADDITION: ret = select_greedy(caching_app, np, opts); break;
            case D2_SAMPLING: ret = select_d2(caching_app, np, opts);         break;
        }
        auto sampled_finish = std::chrono::high_resolution_clock::now();
        std::cerr << "Sampling took " << dur2ms(select_start, sampled_finish) << "ms\n";
        blz::DM<float> distmat(std::get<0>(ret).size(), std::get<1>(ret).size());
        distmat = std::numeric_limits<float>::max();
        minocore::shared::flat_hash_map<uint32_t, uint32_t> centers;
        for(unsigned i = 0; i < std::get<0>(ret).size(); ++i)
            centers.emplace(std::get<0>(ret)[i], i);
#if OLD_CACHER
        for(const auto &pair: caching_app.map_) {
            auto lhi = decltype(caching_app)::KeyType::lh(pair.first);
            auto rhi = decltype(caching_app)::KeyType::rh(pair.first);
            typename minocore::shared::flat_hash_map<uint32_t, uint32_t>::const_iterator it;
            if((it = centers.find(lhi)) != centers.end()) {
                if(it->second < rhi)
                    distmat(it->second, rhi) = pair.second;
                else
                    distmat(rhi, it->second) = pair.second;
            } else if((it = centers.find(rhi)) != centers.end()) {
                if(it->second < lhi)
                    distmat(it->second, lhi) = pair.second;
                else
                    distmat(lhi, it->second) = pair.second;
            }
        }
#else
        std::cerr << "Dispatching threads for mapping map back to matrix\n";
        std::vector<std::thread> threads;
        for(size_t ci = 0; ci < std::get<0>(ret).size(); ++ci) {
            auto center_id = std::get<0>(ret)[ci];
            if(auto it = caching_app.map_.find(center_id); it != caching_app.map_.end()) {
                row(distmat, ci, blaze::unchecked) = it->second;
            } else {
                threads.emplace_back([&,np](size_t ci, size_t center_id) {
                    auto r = row(distmat, ci, blaze::unchecked);
                    for(size_t i = 0; i < np; ++i) {
                        r[i] = app(center_id, i);
                    }
                }, ci, center_id);
            }
        }
        std::fprintf(stderr, "Spun off %zu threads to compute missing rows\n", threads.size());
        for(auto &t: threads) t.join();
#endif
        auto remapped_finish = std::chrono::high_resolution_clock::now();
#if OLD_CACHER
        std::cerr << "Moving hashmap entries to distmat took" << dur2ms(remapped_finish, sampled_finish) << "ms\n";
        for(size_t fi = 0; fi < std::get<0>(ret).size(); ++fi) {
            const auto fid = std::get<0>(ret)[fi];
            const size_t ncities = std::get<1>(ret).size();
            OMP_PFOR
            for(size_t ci = 0; ci < ncities; ++ci) {
                if(distmat(fi, ci) != std::numeric_limits<float>::max())
                    continue;
                distmat(fi, ci) = app(fid, ci);
            }
        }
#else
#endif
        auto dm_finish = std::chrono::high_resolution_clock::now();
        std::cerr << "Calculating remaining entries for distmat took" << dur2ms(dm_finish, remapped_finish)<< "ms\n";
        std::cerr << "Selection took " << dur2ms(dm_finish, select_start) << "ms\n";
        std::get<3>(ret) = std::move(distmat);
    }
    return ret;
}
#endif /* #if 0 */

} // namespace clustering

} // namespace minocore

#endif
