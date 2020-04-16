#include "minocore/bicriteria.h"
#include <iostream>

int main(int argc, char **argv) {
    size_t d = argc == 1 ? 10000: std::atoi(argv[1]);
    unsigned k = argc <= 2 ? 5: std::atoi(argv[2]);
    blaze::DynamicMatrix<double> mat(d, d);
    randomize(mat);
    mat = (abs(mat) * 50) + 1.;
    blaze::band<0>(mat) = 0.;
    std::cerr << submatrix(mat, 0, 0, 25, 25) << '\n';
    std::cerr.flush();
    auto [centers, costs, assignments] = minocore::thorup::oracle_thorup_d(mat, d, k);
    std::fprintf(stderr, "Original oracle thorup D, one iteration\n");
    auto [itercenters, itercosts, iterassignments] = minocore::thorup::iterated_oracle_thorup_d(mat, d, k, 3, 5, (float *)nullptr);
    std::sort(itercenters.begin(), itercenters.end());
    std::fprintf(stderr, "Center set of size %zu has cost %0.12g.\n", itercenters.size(), blz::sum(blz::min<blz::columnwise>(rows(mat, itercenters))));

}
