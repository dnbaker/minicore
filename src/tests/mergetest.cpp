#undef NDEBUG
#include "minicore/util/merge.h"
#include "blaze/Math.h"
using namespace minicore;
int main() {
    blaze::CompressedVector<double> x({1, .0, 1, 1, 1, 0.});
    blaze::CompressedVector<double> y({1, .0, 1, 1, 1, 0.});
    blaze::CompressedVector<double> z({0,1,0,0,0,1});
    blaze::CompressedVector<double> allz({0,0,0,0,0,0});
    blaze::CompressedVector<double> allz2({0,0,0,0,0,0});
    assert(merge::for_each_by_case(6, x.begin(), x.end(), y.begin(), y.end(), [](auto, auto,auto) {}, [](auto, auto){},[](auto,auto){}) == 2);
    assert(merge::for_each_by_case(6, x.begin(), x.end(), z.begin(), z.end(), [](auto, auto,auto) {}, [](auto, auto){},[](auto,auto){}) == 0);
    assert(merge::for_each_by_case(6, x.begin(), x.end(), allz.begin(), allz.end(), [](auto, auto,auto) {}, [](auto, auto){},[](auto,auto){}) == 2);
    assert(merge::for_each_by_case(6, allz2.begin(), allz2.end(), allz.begin(), allz.end(), [](auto, auto,auto) {}, [](auto, auto){},[](auto,auto){}) == 6);
    blaze::CompressedVector<double> allzbut1({0,0,0,0,0,1});
    assert(merge::for_each_by_case(6, allzbut1.begin(), allzbut1.end(), allz.begin(), allz.end(), [](auto, auto,auto) {}, [](auto, auto){},[](auto,auto){}) == 5);
    blaze::CompressedVector<double> allzbut1f({1,0,0,0,0,0});
    assert(merge::for_each_by_case(6, allzbut1f.begin(), allzbut1f.end(), allz.begin(), allz.end(), [](auto, auto,auto) {}, [](auto, auto){},[](auto,auto){}) == 5);
    assert(merge::for_each_by_case(6, allzbut1.begin(), allzbut1.end(), allzbut1f.begin(), allzbut1f.end(), [](auto, auto,auto) {}, [](auto, auto){},[](auto,auto){}) == 4);
    blaze::CompressedVector<double> stripe1({1,0,1,0,1,0});
    blaze::CompressedVector<double> stripe2({0,2,0,2,0,2});
    assert(merge::for_each_by_case(6, stripe1.begin(), stripe1.end(), stripe2.begin(), stripe2.end(), [](auto, auto,auto) {}, [](auto, auto){},[](auto,auto){}) == 0);
    blaze::CompressedVector<double> bandstripe1({1,0,0,1,0,0});
    blaze::CompressedVector<double> bandstripe2({0,2,0,0,1,0});
    assert(merge::for_each_by_case(6, bandstripe1.begin(), bandstripe1.end(), bandstripe2.begin(), bandstripe2.end(), [](auto, auto,auto) {}, [](auto, auto){},[](auto,auto){}) == 2);
    {
        blaze::CompressedVector<double> x({1,0,0,1,0,0});
        blaze::CompressedVector<double> y({0,0,0,0,1,0});
        assert(merge::for_each_by_case(6, x.begin(), x.end(), y.begin(), y.end(), [](auto, auto,auto) {}, [](auto, auto){},[](auto,auto){}) == 3);
    }
}
