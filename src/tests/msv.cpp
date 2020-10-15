#include "include/minicore/util/merge.h"
#include "blaze/Math.h"
#include "blaze/util/SmallArray.h"
#include <random>
#include <set>
#include <cassert>

using namespace minicore;

int main(int argc, char **argv) {

    size_t ndim = argc == 1 ? 100: std::atoi(argv[1]);
    unsigned int nnz = argc <= 2 ? 20: std::atoi(argv[2]);
    std::mt19937_64 mt(argc <= 3 ? 0: std::atoi(argv[3]));
    blaze::DynamicVector<double> lhs = blaze::generate(nnz, [](auto x) {
        return std::pow((x * x * 1337) % 50 - 8.7, 2.);
    }), rhs =  blaze::generate(nnz, [](auto x) {
        return std::pow(((50 - x) * (50 - x) * 1337) % 100 - 80.3, 2);
    });
    blaze::CompressedVector<double> clhs(ndim, nnz), crhs(ndim, nnz);
    auto makear = [&]() {
        blaze::SmallArray<unsigned, 10> ret;
        while(ret.size() < nnz) {
            auto item = mt() % ndim;
            if(std::find(ret.begin(), ret.end(), item) == ret.end())
                ret.pushBack(item);
        }
        std::sort(ret.begin(), ret.end());
        return ret;
    };
    const auto sal = makear(), sar = makear();
    for(size_t i = 0; i < nnz; ++i) {
        clhs.append(sal[i], lhs[i]);
        crhs.append(sar[i], rhs[i]);
    }
    for(size_t i = 0; i < nnz; ++i) {
        assert(clhs[sal[i]] == lhs[i]);
        assert(crhs[sar[i]] == rhs[i]);
    }
    std::set<int> lhi, rhi;
    for(const auto &pair: clhs) lhi.insert(pair.index());
    for(const auto &pair: crhs) rhi.insert(pair.index());
    std::set<int> unionset;
    unionset.merge(lhi); unionset.merge(rhi);
    auto sharedfunc =
        [&](auto idx, auto x, auto y) {
#if 0
            std::fprintf(stderr, "Index %zu has %g/%g (both nonzero) for x and y, respectively.\n", idx, double(x), double(y));
#else
            assert(clhs[idx] == x);
            assert(crhs[idx] == y);
#endif
        };
    auto lhfunc =
        [&](auto idx, auto x) {assert(clhs[idx] == x);};
    auto rhfunc =
        [&](auto idx, auto y) {assert(crhs[idx] == y);};
    size_t sharednz = merge::for_each_by_case(ndim, clhs.begin(), clhs.end(), crhs.begin(), crhs.end(), sharedfunc, lhfunc, rhfunc);
    assert(sharednz == ndim - unionset.size());
}
