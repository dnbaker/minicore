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
    r1[0] = r0[0] = 0.;
    r1 *= r1;
    r0 *= r0;
    r1 /= l2Norm(r1);
    r0 /= l2Norm(r0);
    std::fprintf(stderr, "multinomial jsd: %f\n", distance::multinomial_jsd(r1, r0));
}
