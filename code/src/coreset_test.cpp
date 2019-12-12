#include "include/coreset.h"

int main() {
    coresets::Coreset<uint32_t, float> cs(100);
    coresets::CoresetSampler<float, uint32_t> sampler;
    size_t npoints = 10000;
    std::vector<uint32_t> assignments(npoints);
    std::vector<float>          costs(npoints);
    size_t ncenters = 20;
    for(auto &v: assignments) v = std::rand() % ncenters;
    for(auto &v: costs)       v = 1. / std::pow(float(std::rand()) / RAND_MAX, 3);
    sampler.make_sampler(npoints, ncenters, costs.data(), assignments.data() /* weights = nullptr */);
    std::vector<float> weights(npoints);
    for(auto &v: weights)       v = 1. / std::pow(float(std::rand()) / RAND_MAX, 2);
    sampler.make_sampler(npoints, ncenters, costs.data(), assignments.data(), weights.data(), 2);
    auto sample = sampler.sample(20);
    std::fprintf(stderr, "sample of 20 is of size %zu\n", sample.size());
    //if(0) sampler.make_sampler(10, 10, nullptr, nullptr);
}
