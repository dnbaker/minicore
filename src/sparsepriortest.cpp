#include "minocore/dist/applicator.h"

int main() {
    blaze::CompressedMatrix<double> cm{{1., 5., 0., 3., 1., 1., 1., 3., 1., 1}, { 1.,  1.,  3.,  2.,  2.,  0., 21.,  1.,  7.,  1. }};
    std::cerr << cm << '\n';
    auto app = minocore::make_probdiv_applicator(cm, blz::JSD, minocore::jsd::DIRICHLET);
    assert(std::abs(app(0, 1) - 0.16066042325849725) < 1e-5);
    //std::fprintf(stderr, "with dirichlet: %g.\n", app(0, 1));
}
