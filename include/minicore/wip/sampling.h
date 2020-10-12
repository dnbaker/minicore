#ifndef CLUSTERING_SAMPLING_H__
#define CLUSTERING_SAMPLING_H__
#include "minicore/clustering/traits.h"
#include "minicore/optim/oracle_thorup.h"

namespace minicore {

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

} // namespace clustering

} // namespace minicore

#endif
