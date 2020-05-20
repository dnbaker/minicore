#include "minocore/dist/applicator.h"

int main() {
    blaze::CompressedMatrix<double> cm{{1., 5., 0., 3., 1., 1., 1., 3., 1., 1}, { 1.,  1.,  3.,  2.,  2.,  0., 21.,  1.,  7.,  1. }};
    std::cerr << cm << '\n';
    auto r1 = row(cm, 0);
    auto r2 = row(cm, 1);
    for(auto &pair: r1) std::fprintf(stderr, "v %g at %zu\n", pair.value(), pair.index());
    auto app = minocore::make_probdiv_applicator(cm, blz::JSD, minocore::jsd::DIRICHLET);
    assert(std::abs(app(0, 1) - 0.16066042325849725) < 1e-5);
    std::fprintf(stderr, "with dirichlet: %g.\n", app(0, 1));
    std::fprintf(stderr, "without dirichlet: %g.\n", minocore::make_probdiv_applicator(cm, blz::JSD)(0, 1));
}
