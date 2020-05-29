#include "minocore/coreset.h"

using FT = float;
using namespace cs;
int main() {
    CoresetSampler<FT> cs;
    std::vector<float> costs(100);
    std::vector<unsigned> asn(100);
    for(auto &i: asn) i = std::rand() % 5;
    for(auto &i: costs) i = float(std::rand()) / RAND_MAX;
    cs.make_sampler(100, 5, costs.data(), asn.data());
    cs.write("test_io.cs");
    CoresetSampler<FT> cs2("test_io.cs");
    assert(cs == cs2);
    std::system("rm test_io.cs");
}

