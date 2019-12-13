#include "kmeans.h"

int main() {
    std::vector<std::vector<float>> stuff;
    wy::WyRand<uint32_t, 2> gen;
    for(auto i = 0; i < 20; ++i) {
        stuff.emplace_back(100);
        for(auto &e: stuff.back()) e = double(std::rand()) / RAND_MAX;
    }
    auto centers = clustering::kmeanspp(stuff.begin(), stuff.end(), gen, 13);
}
