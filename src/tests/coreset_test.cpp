#include "minicore/coreset.h"
using namespace minicore;

int main() {
    coresets::IndexCoreset<uint32_t, float> cs(100);
    coresets::CoresetSampler<float, uint32_t> sampler;
    size_t npoints = 10000;
    std::vector<uint32_t> assignments(npoints);
    std::vector<float>          costs(npoints);
    size_t ncenters = 20;
    for(auto &v: assignments) v = std::rand() % ncenters;
    for(auto &v: costs)       v = std::rand() % 3;
    sampler.make_sampler(npoints, ncenters, costs.data(), assignments.data() /* weights = nullptr */);
    std::vector<float> weights(npoints);
    for(auto &v: weights)       v = 1. / ((std::rand() % 5) + 1);
    sampler.make_sampler(npoints, ncenters, costs.data(), assignments.data(), weights.data(), 2);
    auto sample = sampler.sample(20);
    sample.show();
    std::fprintf(stderr, "sample of 20 is of size %zu\n", sample.size());
#if 0
    sample.show();
    sample.show();
    sample.show();
#endif
    //sample.show();
    sample.compact();
    std::fprintf(stderr, "sample of 20 is of size %zu after compacting\n", sample.size());
    coresets::CoresetSampler<float, uint32_t> sampler2;
    sampler2.make_sampler(npoints, ncenters, costs.data(), assignments.data(), (double *)nullptr, coresets::LBK);
    sampler2.sample(20);
    //if(0) sampler.make_sampler(10, 10, nullptr, nullptr);
}
