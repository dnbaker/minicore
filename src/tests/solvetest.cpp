#include "include/minocore/clustering/solve.h"
#include "src/tests/solvetestdata.cpp"

namespace clust = minocore::clustering;
using namespace minocore;

template<typename LH, typename RH>
INLINE double compute_cost(const LH &lh, const RH &rh, double psum, double pv) {
    auto lhsum = blz::sum(lh) + psum, rhsum = blz::sum(rh) + psum;
    auto lhi = 1. / lhsum, rhi = 1. / rhsum;
    auto lhb = lh.begin(), lhe = lh.end();
    auto rhb = rh.begin(), rhe = rh.end();
    double ret = 0.;
    auto func = [&](auto xv, auto yv) ALWAYS_INLINE {
        xv *= lhi; yv *= rhi;
        assert(xv < 1.);
        assert(yv < 1.);
        auto mni = 2. / (xv + yv);
        return (xv * std::log(xv * mni) + yv * std::log(yv * mni));
    };
    auto sharednz = minocore::merge::for_each_by_case(
        lh.size(), lhb, lhe, rhb, rhe,
        [&,pv](auto,auto x, auto y) {ret += func(x + pv, y + pv);},
        [&,pv](auto,auto x) {ret += func(x + pv, pv);},
        [&,pv](auto,auto y) {ret += func(pv, y + pv);});
    if(lhsum != rhsum)
        ret += func(pv, pv) * sharednz;
    ret *= .5; // account for .5
    ret = std::max(ret, 0.);
    return ret;
}

int main(int argc, char *argv[]) {
    dist::print_measures();
    if(std::find_if(argv, argc + argv, [](auto x) {return std::strcmp(x, "-h") == 0;}) != argc + argv)
        std::exit(1);
    const size_t nr = x.rows(), nc = x.columns();
    blz::DV<double> prior{1. / nc};
    dist::DissimilarityMeasure msr = dist::MKL;
    if(argc > 1) {
        msr = (dist::DissimilarityMeasure)std::atoi(argv[1]);
        std::fprintf(stderr, "This may not work if you change the measure but not the original costs\n");
    }
    std::fprintf(stderr, "msr: %d/%s\n", (int)msr, dist::msr2str(msr));
    std::vector<blaze::CompressedVector<double, blaze::rowVector>> centers;
    std::vector<int> ids{1018, 2624, 5481, 6006, 8972};
    while(ids.size() < 10) {
        auto rid = std::rand() % x.rows();
        if(std::find(ids.begin(), ids.end(), rid) == ids.end())
            ids.emplace_back(rid);
    }
    for(const auto id: ids) centers.emplace_back(row(x, id));
    const size_t k = centers.size();
    const double psum = prior[0] * nc, pv = prior[0];
    blz::DV<uint32_t> asn(nr);
    blz::DV<double> hardcosts = blaze::generate(nr, [&](auto id) {
        auto r = row(x, id);
        uint32_t bestid = 0;
        double ret = compute_cost(centers[0], r, psum, pv);
        for(unsigned j = 1; j < k; ++j) {
            auto x = compute_cost(centers[j], r, psum, pv);
            if(x < ret) ret = x, bestid = j;
        }
        assert(id < asn.size());
        asn[id] = bestid;
        return ret;
    });
    auto mnc = blz::min(hardcosts);
    std::fprintf(stderr, "Total cost: %g. max cost: %g. min cost: %g. mean cost:%g\n", blz::sum(hardcosts), blz::max(hardcosts), mnc, blz::mean(hardcosts));
    std::vector<uint32_t> counts(k);
    for(const auto v: asn) ++counts[v];
    for(unsigned i = 0; i < k; ++i) {
        std::fprintf(stderr, "Center %d with sum %g has %u supporting, with total cost of assigned points %g\n", i, blz::sum(centers[i]), counts[i],
                     blz::sum(blz::generate(nr, [&](auto id) { return asn[id] == i ? hardcosts[id]: 0.;})));
    }
    assert(min(asn) == 0);
    assert(max(asn) == centers.size() - 1);
    clust::perform_hard_clustering(x, msr, prior, centers, asn, hardcosts);
    //minocore::perform_soft_clustering(x, minocore::distance::JSD, prior, centers, costs);
}
