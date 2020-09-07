#include "minocore/util/csc.h"
#include <iostream>
#undef NDEBUG
#include <cassert>

int main() {
    size_t np = 100, ndim = 10000;
    blz::DV<double> f1 = blaze::generate(np, [](auto) {return std::ldexp(std::rand(), -31);});
    blz::DV<int>    i1 = blaze::generate(np, [](auto x) {return x * x + 1;});
    std::cerr << trans(f1);
    std::cerr << trans(i1);
    assert(max(i1) < int(ndim));
    auto cv = minocore::util::make_csparse_view(f1.data(), i1.data(), np, ndim);
    std::cerr << "cv made of size " << cv.size() << " with " << cv.nnz() << '\n';
    auto eit = cv.end();
    auto it = cv.begin();
    assert(it.col_.data_ == f1.data());
    assert(it.col_.indices_ == i1.data());
    for(;it != eit; ++it) {
        std::cerr << it->second << '\n';
    }
    size_t idx = 0;
    for(const auto &pair: cv) {
        assert(f1[idx] == pair.value());
        assert(i1[idx] == int(pair.index()));
        ++idx;
    }
}
