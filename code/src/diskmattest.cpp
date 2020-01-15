#include "fgc/diskmat.h"
#include "fgc/distance.h"

using namespace fgc;
using namespace blz;

int main() {
    const char *fn = "./zomg.dat";
    {
        DiskMat<float> dm(200, 1000, fn);
        dm.delete_file_ = true;
        assert(::access(fn, F_OK) != -1);
    }
    assert(::access(fn, F_OK) == -1);
    DiskMat<float> dm(200, 1000, fn);
    dm.delete_file_ = true;
    ~dm = 0;
    assert(dm(1, 4) == 0.);
    assert(blaze::sum(~dm) == 0);
    auto r1 = row(dm, 1), r0 = row(dm, 0);
    randomize(r1);
    randomize(r0);
    r1[0] = 0.;
    r0[0] = 0.;
    r1[4] = 0.;
    r1 *= r1;
    r0 *= r0;
    r1 /= l2Norm(r1);
    r0 /= l2Norm(r0);
    blaze::CompressedVector<float> c1(r0.size()), c0(r0.size());
    c1.reserve(r1.size() - 2);
    c0.reserve(r0.size() - 1);
    for(size_t i = 0; i < r0.size(); ++i) {
        if(r0[i])
            c0.append(i, r0[i]);
        if(r1[i])
            c1.append(i, r1[i]);
    }
    std::fprintf(stderr, "multinomial jsd: %f\n", distance::multinomial_jsd(r1, r0));
    std::fprintf(stderr, "multinomial jsd: %f\n", distance::multinomial_jsd(c1, c0));
}
