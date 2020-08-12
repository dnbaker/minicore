#include "include/minocore/clustering/solve.h"
#include "src/tests/solvetestdata.cpp"

namespace clust = minocore::clustering;
using namespace minocore;

// #define double float

int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false);
    dist::print_measures();
    if(std::find_if(argv, argc + argv, [](auto x) {return std::strcmp(x, "-h") == 0;}) != argc + argv)
        std::exit(1);
    const size_t nr = x.rows(), nc = x.columns();
    blz::DV<double> prior{double(1. / nc)};
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
    const double psum = prior[0] * nc;
    blz::DV<uint32_t> asn(nr);
    blz::DV<double> hardcosts = blaze::generate(nr, [&](auto id) {
        auto r = row(x, id);
        uint32_t bestid = 0;
        double ret = cmp::msr_with_prior(msr, r, centers.front(), prior, psum);
        for(unsigned j = 1; j < k; ++j) {
            double x = cmp::msr_with_prior(msr, r, centers[j], prior, psum);
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
