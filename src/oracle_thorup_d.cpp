#include "fgc/bicriteria.h"

int main(int argc, char **argv) {
    size_t d = argc == 1 ? 10000: std::atoi(argv[1]);
    unsigned k = argc <= 2 ? 5: std::atoi(argv[2]);
    blaze::DynamicMatrix<float> mat(d, d);
    randomize(mat);
    mat = abs(mat);
    auto [centers, costs, assignments] = fgc::thorup::oracle_thorup_d(mat, d, k);
    auto [itercenters, itercosts, iterassignments] = fgc::thorup::iterated_oracle_thorup_d(mat, d, k, 8, 5, (float *)nullptr, 7);
}
