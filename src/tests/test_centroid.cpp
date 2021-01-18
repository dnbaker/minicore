#undef NDEBUG
#include "minicore/clustering/centroid.h"

using namespace minicore::clustering;

namespace mc = minicore;
int test3() {
    const size_t n = 10;
    std::vector<double> cd(n);
    for(auto &i: cd) i = 1.3;
    std::vector<uint32_t> ci(n);
    for(unsigned int i = 0; i < n; ++i) ci[i] = i * 2;
    blaze::CompressedVector<double> cv(n);
    mc::util::CSparseVector<double, uint32_t> csv(cd.data(), ci.data(), n, 100);
    blaze::DynamicVector<double> dv(100);
    dv = 0;
    assert(sum(dv) == 0.);
    for(size_t i = 0; i < n; ++i) {
        dv[ci[i]] = cd[i];
    }
    blaze::DynamicVector<double> dv2;
    dv2.resize(100);
    mc::clustering::set_center(dv2, csv);
    assert(dv2 == dv);
    return 0;
}

int test1() {
    const size_t n = 10;
    std::vector<double> cd(n);
    for(auto &i: cd) i = 1.3;
    std::vector<uint32_t> ci(n);
    for(unsigned int i = 0; i < n; ++i) ci[i] = i * 2;
    mc::util::CSparseVector<double, uint32_t> csv(cd.data(), ci.data(), n, 100);
    blaze::DynamicVector<double> dv(100);
    dv = 0;
    assert(sum(dv) == 0.);
    for(size_t i = 0; i < n; ++i) {
        dv[ci[i]] = cd[i];
    }
    blaze::DynamicVector<double> dv2;
    dv2.resize(100);
    std::fprintf(stderr, "Resized\n");
    mc::clustering::set_center(dv2, csv);
    assert(dv2 == dv);
    return 0;
}

int test2() {
    blaze::DynamicMatrix<double> dm(10, 10);
    dm = 0;
    for(size_t i = 0; i < 10; ++i) dm(i, i) = 10.;
    blaze::DynamicVector<double, blz::rowVector> dv(10);
    blaze::DynamicVector<uint32_t> indices = blaze::generate(10, [](auto x) {return x;});
    mc::clustering::set_center(dv, dm, indices.data(), 10);
    std::cerr << dv << '\n';
    assert(std::all_of(dv.begin(), dv.end(), [](auto x) {return x == 1.;}));
    dv = 1.3;
    blaze::CompressedMatrix<double> sm = dm;
    blaze::DynamicVector<double> data = blaze::generate(10, [](auto) {return 1.;});
    blaze::DynamicVector<uint32_t> indptr(11);
    for(size_t i = 0; i < 10; ++i) {
        indptr[i + 1] = indptr[i] + 1;
    }
    std::fprintf(stderr, "testing csm\n");
    for(size_t i = 0; i < 10; ++i) {
        std::fprintf(stderr, "%g:%u:%u-%u\n", data[i], indices[i], indptr[i], indptr[i + 1]);
    }
    mc::util::CSparseMatrix<double, uint32_t, uint32_t> csm(data.data(), indices.data(), indptr.data(), 10, 10, 10);
    blaze::DynamicVector<uint32_t> udata = blaze::generate(10, [](auto) {return 1;});
    blaze::DynamicVector<float> fdata = blaze::generate(10, [](auto) {return 1.;});
    mc::util::CSparseMatrix<uint32_t, uint32_t, uint32_t> ucsm(udata.data(), indices.data(), indptr.data(), 10, 10, 10);
    mc::util::CSparseMatrix<double, uint32_t, uint32_t> fcsm(data.data(), indices.data(), indptr.data(), 10, 10, 10);
    mc::clustering::set_center(dv, csm, indices.data(), 10);
    std::cerr << dv << '\n';
    assert(std::all_of(dv.begin(), dv.end(), [](auto x) {return std::abs(x - .1) < 1e-8;}));
    dv = -1.;
    std::fprintf(stderr, "Try unsigned matrix\n");
    mc::clustering::set_center(dv, ucsm, indices.data(), 10);
    std::cerr << dv << '\n';
    assert(std::all_of(dv.begin(), dv.end(), [](auto x) {std::fprintf(stderr, "x: %.20g. dist: %g\n", double(x), std::abs(x - .1)); return std::abs(x - .1) < 1e-8;}));
    std::fprintf(stderr, "Now setting blaze matrix\n");
    dv = -1.;
    blaze::CompressedMatrix<double> cm(10, 10);
    cm.reserve(10);
    for(size_t i = 0; i < 10; ++i) cm(i,i) = 1.;
    mc::clustering::set_center(dv, cm, indices.data(), 10);
    std::cerr << dv << '\n';
    assert(std::all_of(dv.begin(), dv.end(), [](auto x) {return std::abs(x - .1) < 1e-8;}));
    dv = -1.;
    std::fprintf(stderr, "Try float matrix\n");
    mc::clustering::set_center(dv, fcsm, indices.data(), 10);
    std::cerr << dv << '\n';
    assert(std::all_of(dv.begin(), dv.end(), [](auto x) {std::fprintf(stderr, "x: %.20g. dist: %g\n", double(x), std::abs(x - .1)); return std::abs(x - .1) < 1e-8;}));
    dv = -1.;
    blaze::DynamicVector<uint32_t> indices2 = blaze::generate(50, [](auto x) {return 2 * (x % 5);});
    fdata = blaze::generate(50, [](auto x) {return 2. * (x + 1.);});
    std::cerr << fdata << '\n';
    indptr = blaze::generate(11, [](auto x) {return x * 5;});
    mc::util::CSparseMatrix<float, uint32_t, uint32_t> fcsm2(fdata.data(), indices2.data(), indptr.data(), 10, 10, 50);
    for(size_t i = 0; i < 10; ++i) {
        auto r = row(fcsm2, i);
        std::cerr << r << '\n';
    }
    std::cerr << fcsm2 << '\n';
    set_center(dv, fcsm2, indices.data(), 10);
    std::cerr << dv << '\t';
    for(size_t i = 0; i < 5; ++i) {
        std::fprintf(stderr, "dvi: %g. rhi: %g\n", double(dv[i * 2]), double(47 + i * 2));
        assert(size_t(dv[i * 2]) == size_t(47 + i * 2));
    }
    blaze::DynamicMatrix<float> dm2(500, 5);
    dm2 = 0;
    column(dm2, 0) = 1.;
    column(dm2, 3) = -1.4;
    blz::DV<float,blz::rowVector> ctr;
    indices2 = blaze::generate(500, [](auto x) {return x;});
    set_center(ctr, dm2, indices2.data(), 500);
    dm2 = 0;
    for(size_t i = 0; i < 500; ++i) {
        for(size_t j = 0; j < 5; ++j) {
            dm2(i, j) = (j + 1) * (i + 1);
        }
    }
    std::cerr << "Set d2!\n" << dm2 << '\n';
    const auto rowsums = blaze::generate(500, [&](auto x){return sum(row(dm2, x));});
    set_center(ctr, dm2, indices2.data(), 500, static_cast<blz::DV<double> *>(nullptr), &rowsums);
    std::cerr << ctr << '\n';
    return 0;
}

int main() {
    return test1() || test2() || test3();
}
