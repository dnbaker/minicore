#include "include/coreset.h"

int main() {
    coresets::Coreset<uint32_t, float> cs(100);
    coresets::CoresetSampler<float, uint32_t> sampler;
    if(0) sampler.make_sampler(10, 10, nullptr, nullptr);
}
