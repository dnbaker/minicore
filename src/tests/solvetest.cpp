#include "include/minocore/clustering/solve.h"
#include "src/tests/solvetestdata.cpp"

namespace clust = minocore::clustering;

template<typename LH, typename RH>
INLINE double compute_cost(const LH &lh, const RH &rh, double psum, double pv) {
    auto lhsum = blz::sum(lh) + psum, rhsum = blz::sum(rh) + psum;
    auto lhi = 1. / lhsum, rhi = 1. / rhsum;
    auto lhb = lh.begin(), lhe = lh.end();
    auto rhb = rh.begin(), rhe = rh.end();
    double ret = 0.;
    auto func = [&](auto xv, auto yv) {
        xv *= lhi; yv *= rhi;
        auto mni = 2. / (xv + yv);
        return (xv * std::log(xv * mni) + yv * std::log(yv * mni));
    };
    auto sharednz = minocore::merge::for_each_by_case(
        lh.size(), lhb, lhe, rhb, rhe,
        [&,pv](auto,auto x, auto y) {ret += func(x + pv, y + pv);},
        [&,pv](auto,auto x) {ret += func(x + pv, pv);},
        [&,pv](auto,auto y) {ret += func(pv, y + pv);});
    ret += func(pv, pv) * sharednz;
    ret *= .5; // account for .5
    return ret;
}

int main() {
    const size_t nr = x.rows(), nc = x.columns();
    blz::DV<double> prior{.1};
    std::vector<blaze::CompressedVector<double, blaze::rowVector>> centers;
    for(const auto id: {1018, 2624, 5481, 6006, 8972})
        centers.emplace_back(row(x, id));
    const size_t k = centers.size();
    const double psum = prior[0] * nc, pv = prior[0];
    blz::DV<uint32_t> asn(nr);
    blz::DV<double> hardcosts = blaze::generate(nr, [&,k](auto id) {
        auto r = row(x, id);
        uint32_t bestid = 0;
        double ret = compute_cost(centers[0], r, psum, pv);
        for(unsigned j = 1; j < k; ++j) {
            auto x = compute_cost(centers[j], r, psum, pv);
            if(x < ret) ret = x, bestid = j;
        }
        asn[id] = bestid;
        return ret;
    });
    clust::perform_hard_clustering(x, dist::JSD, prior, centers, asn, hardcosts);
    //minocore::perform_soft_clustering(x, minocore::distance::JSD, prior, centers, costs);
}
