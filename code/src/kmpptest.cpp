#include "kmeans.h"
#include <chrono>
#define t std::chrono::high_resolution_clock::now

int main() {
    std::vector<std::vector<float>> stuff;
    wy::WyRand<uint32_t, 2> gen;
    for(auto i = 0; i < 200; ++i) {
        stuff.emplace_back(100);
        for(auto &e: stuff.back()) e = double(std::rand()) / RAND_MAX;
    }
    auto start = t();
    auto centers = clustering::kmeanspp(stuff.begin(), stuff.end(), gen, 13);
    auto stop = t();
    std::fprintf(stderr, "Time: %zu\n", size_t((stop - start).count()));
}
