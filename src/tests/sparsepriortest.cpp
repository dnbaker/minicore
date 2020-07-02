#include "minocore/dist/applicator.h"

int main() {
    blaze::CompressedMatrix<double> cm{{1., 5., 0., 3., 1., 1., 1., 3., 1., 1}, { 1.,  1.,  3.,  2.,  2.,  0., 21.,  1.,  7.,  1. }};
    std::cerr << cm << '\n';
    auto app = minocore::make_probdiv_applicator(cm, blz::JSD, minocore::jsd::DIRICHLET);
    assert(std::abs(app(0, 1) - 0.16066042325849725) < 1e-10 || !std::fprintf(stderr, "got %g vs %g\n", app(0, 1), 0.16066042325849725));
    blaze::CompressedMatrix<double> cm2{
        {0, 7, 6, 0, 6, 6, 0, 0, 7, 9, 4, 0, 0, 0, 6, 6, 0, 0, 0, 7},
        {6, 7, 0, 0, 0, 5, 6, 9, 0, 0, 0, 0, 0, 9, 0, 6, 5, 6, 0, 0}
    };
    auto r1 = row(cm2, 0);
    auto r2 = row(cm2, 1);
    assert(blz::number_shared_zeros(r1, r2) == 4);
    auto app2 = minocore::make_probdiv_applicator(cm2, blz::JSD, minocore::jsd::DIRICHLET);
    double v2 = app2(0, 1);
    static constexpr double correct2 = 0.2307775339934756;
    assert(std::abs(correct2 - v2) < 1e-10);
    auto app3 = minocore::make_probdiv_applicator(cm2, blz::MKL, minocore::jsd::DIRICHLET);
    app3(0, 1);
    //std::fprintf(stderr, "difference: %0.12g\n", correct2 - v2);
}
