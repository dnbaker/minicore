//#define VERBOSE_AF 1
#include "minocore/optim/lsearch.h"
#include "minocore/util/diskmat.h"
#include <iostream>

using namespace minocore;

int main(int argc, char *argv[]) {
    unsigned n = argc == 1 ? 100: std::atoi(argv[1]);
    unsigned k = 20;
    double eps = 0.01;
    Graph<> g(n);
    for(unsigned i = 0; i < n - 1; ++i) {
        boost::add_edge(i, i + 1, 2.5, g);
    }
    for(unsigned i = 0; i < n; ++i) {
        boost::add_edge(i, std::rand() % n, double(std::rand()) / RAND_MAX, g);
        boost::add_edge(i, std::rand() % n, double(std::rand()) / RAND_MAX, g);
        boost::add_edge(i, std::rand() % n, double(std::rand()) / RAND_MAX, g);
    }
    auto dm = graph2diskmat(g, "./zomg.dat");
    //std::cout << ~dm << '\n';
    dm.delete_file_ = true;
    auto lsearcher = make_kmed_lsearcher(~dm, k, eps);
    lsearcher.run();

    std::vector<float> weights(n);
    wy::WyHash<uint32_t, 2> rng(13);
    std::uniform_real_distribution<float> vals(std::nextafter(0., 17.), 17.);
    for(auto p = weights.data(), e = p + n; p < e; *p++ = vals(rng));
    auto wp = weights.data();
    for(auto &w: weights) w *= w;
    DiskMat<float> weighted_dm(dm, nullptr);
    weighted_dm.delete_file_ = true;
    assert(weighted_dm.rows() == dm.rows());
    assert(weighted_dm.columns() == dm.columns());
    const auto nr = weighted_dm.rows();
    //OMP_PRAGMA("omp parallel for")
    for(size_t i = 0; i < nr; ++i)
        row(~weighted_dm, i) *= wp[i];
    auto lsearcher_cp = make_kmed_lsearcher(~weighted_dm, k, eps);
    lsearcher_cp.run();
    auto subm = blaze::submatrix(~dm, 0, 0, 50, n);
    assert(subm.rows() == 50);
    assert(subm.columns() == n);
    blaze::CustomMatrix<float, blaze::aligned, blaze::padded, blaze::rowMajor> cm(subm.data(), subm.rows(), subm.columns(), subm.spacing());
    auto lsearcher_fewer_facilities = make_kmed_lsearcher(cm, k, eps);
    lsearcher_fewer_facilities.run();
}
