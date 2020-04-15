#include "fgc/bicriteria.h"
#include <iostream>

int main(int argc, char **argv) {
    size_t d = argc == 1 ? 10000: std::atoi(argv[1]);
    unsigned k = argc <= 2 ? 5: std::atoi(argv[2]);
    blaze::DynamicMatrix<double> mat(d, d);
    randomize(mat);
    mat = (abs(mat) * 50) + 1.;
    std::cout << submatrix(mat, 0, 0, 25, 25) << '\n';
#if 0
    auto [centers, costs, assignments] = fgc::thorup::oracle_thorup_d(mat, d, k);
    std::fprintf(stderr, "Original oracle thorup D, one iteration\n");
#endif
    auto [itercenters, itercosts, iterassignments] = fgc::thorup::iterated_oracle_thorup_d(mat, d, k, 1, 5, (float *)nullptr, 7);
}
