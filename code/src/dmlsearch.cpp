#include "fgc/lsearch.h"
#include "fgc/diskmat.h"
#include <iostream>

using namespace fgc;

int main() {
    const char *fn = "./zomg.dat";
    unsigned n = 100;
    unsigned k = 10;
    double eps = 0.01;
    Graph<> g(n);
    for(int i = 0; i < n - 1; ++i) {
        boost::add_edge(i, i + 1, 2.5, g);
    }
    for(int i = 0; i < n; ++i) {
        boost::add_edge(i, std::rand() % n, double(std::rand()) / RAND_MAX, g);
        boost::add_edge(i, std::rand() % n, double(std::rand()) / RAND_MAX, g);
        boost::add_edge(i, std::rand() % n, double(std::rand()) / RAND_MAX, g);
    }
    auto dm = graph2diskmat(g, fn);
    std::cout << ~dm << '\n';
    dm.delete_file_ = true;
    std::vector<float> weights(n);
    wy::WyHash<uint32_t, 2> rng(13);
    std::uniform_real_distribution<float> vals(std::nextafter(0., 17.), 17.);
    for(auto p = weights.data(), e = p + n; p < e; *p++ = vals(rng));
    auto wp = weights.data();
    auto lsearcher = make_kmed_lsearcher(~dm, k, eps, wp);
}
