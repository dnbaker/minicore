#include "fgc/lsearch.h"
#include "fgc/diskmat.h"
#include <iostream>

using namespace fgc;

int main() {
    const char *fn = "./zomg.dat";
    Graph<> g(100);
    for(int i = 0; i < 99; ++i) {
        boost::add_edge(i, i + 1, 1.4, g);
    }
    for(int i = 0; i < 100; ++i) {
        boost::add_edge(i, std::rand() % 100, double(std::rand()) / RAND_MAX, g);
    }
    auto dm = graph2diskmat(g, fn);
    std::cout << ~dm << '\n';
    dm.delete_file_ = true;
}
