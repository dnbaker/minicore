#include "minicore/util/blaze_adaptor.h"
#include "minicore/coreset/gmm.h"
#include <iostream>

int main() {
    blz::DynamicMatrix<float> mat(3, 2);
    for(unsigned i = 0; i < mat.rows(); ++i)
        for(unsigned j = 0; j < mat.columns(); ++j)
            mat(i, j) = std::pow(j, i), std::fprintf(stderr, "%u/%u: %f\n", i, j, mat(i, j));
    for(const auto row: mat.rowiterator()) {
        std::cout << row << '\n';
    }
/*
    const blz::DynamicMatrix<float> matcp(mat);
    for(const auto row: matcp.rowiterator()) {
        std::cout << row << '\n';
    }
*/
    for(const auto column: mat.columniterator()) {
        std::cout << column << '\n';
    }
    auto r1 = row(mat, 1);
    auto r2 = row(mat, 2);
    std::cout << r1;
    std::cout << r2;
    std::fprintf(stderr, "pointers: %p, %p\n", (void *)&r1[0], (void *)&r2[0]);
    std::fprintf(stderr, "rdiff norm: %f\n", blz::sqrDist(r1, r2));
    minicore::GMM gm(5, 20);
    if(0) {
        gm.logprob(r1, 1);
        gm.logprob(r2, r1);
    }
    //std::fprintf(stderr, "cdiff norm: %f\n", blz::sqrDist(column(mat, 1), column(mat, 0)));
}
