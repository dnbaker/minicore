#include "include/minocore/clustering/solve.h"

namespace clust = minocore::clustering;

int main() {
    blz::DM<double> x = blaze::generate(10, 100, [](auto x, auto y) {return double(x + y) / (x * y + y);});
    blz::DV<double, blaze::rowVector> y = blaze::generate<blaze::rowVector>(10, [](auto x) {return x;});
    auto prior = y;
    auto hardcosts = evaluate(y + 2);
    blz::DV<uint32_t> asn = blaze::generate(10, [](auto x) {return x + 1;});
    blz::DM<double> costs = blaze::generate(10, 100, [](auto, auto) {return 1;});
    std::vector<blaze::DynamicVector<double, blaze::rowVector>> centers;
    clust::perform_hard_clustering(x, minocore::distance::JSD, prior, centers, asn, hardcosts);
    //minocore::perform_soft_clustering(x, minocore::distance::JSD, prior, centers, costs);
}
