#include "fgc/diskmat.h"

using namespace fgc;

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
}
