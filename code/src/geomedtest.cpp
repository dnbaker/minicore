#include "fgc/kmedian.h"
#include <chrono>

auto t() {return std::chrono::high_resolution_clock::now();}

template<typename Mat, typename Vec>
auto l1dist(const Mat &m, const Vec &vec) {
    double ret = 0.;
    for(const auto r: blz::rowiterator(m)) {
        ret += blz::l1Dist(vec, r);
    }
    return ret;
}
using blz::ElementType_t;
template<typename Mat, typename Vec>
auto l1dists(const Mat &m, const Vec &vec) {
    blz::DV<ElementType_t<Mat>> ret(m.rows());
    auto retit = ret.begin();
    for(const auto r: blz::rowiterator(m)) {
        *retit++ = blz::l1Dist(vec, r);
    }
    return ret;
}

template<typename Mat, typename Vec>
auto reductionl1(const Mat &m, const Vec &vec) {
    return blz::sum(blz::abs(m - blz::expand(vec, m.rows())));
}
template<typename Mat, typename Vec>
auto expandl1s(const Mat &m, const Vec &vec) {
    return blz::DV<ElementType_t<Mat>>(blaze::sum<blz::rowwise>(blz::abs(m - blz::expand(vec, m.rows()))));
}

int main(int c, char **a) {
    unsigned dim = c < 2 ? 2000: std::atoi(a[1]);
    unsigned nr = c < 3 ? 100: std::atoi(a[2]);
    blaze::DynamicMatrix<float> m(nr, dim);
    blaze::setSeed(0);
    randomize(m);
    m = pow(abs(m), -1.2);
    auto sel = rows(m, [](auto i) {return i * 2 + 1;}, 50);
    blaze::DynamicVector<float, blaze::rowVector> v, v2(dim);
    fgc::coresets::geomedian(m, v);
    std::cout << subvector(v, 0, nr/2) << '\n';
    fgc::coresets::l1_median(m, v2);
    std::cout << subvector(v2, 0, nr / 2) << '\n';
    std::cout << subvector(blaze::mean<blaze::columnwise>(m), 0, 50) << '\n';
    auto start = t();
    double cwmed = l1dist(m, v2);
    auto stop = t();
    auto manstart = t();
    double cwmed2 = reductionl1(m, v2);
    auto manstop = t();
    std::fprintf(stderr, "Manual l1 dist time: %zu/%g. reduction-based: %zu/%g\n", size_t((stop - start).count() / 1000), cwmed, size_t((manstop - manstart).count() / 1000), cwmed2);
    start = t();
    auto l1distances = l1dists(m, v2);
    stop = t();
    manstart = t();
    auto l1distances2 = expandl1s(m, v2);
    manstop = t();
    std::fprintf(stderr, "Manual l1 distances time: %zu/%g. reduction-based: %zu/%g\n", size_t((stop - start).count() / 1000), cwmed, size_t((manstop - manstart).count() / 1000), cwmed2);
    std::cout << "L1 dist under geomedian: " << l1dist(m, v) << '\n';
    std::cout << "L1 dist under component-wise median: " << cwmed << '\n';
    std::cout << "L1 dist under component-wise mean: " << l1dist(m, blaze::evaluate(blaze::mean<blaze::columnwise>(m))) << '\n';
    for(unsigned feature = 0; feature < m.rows(); ++feature) {
        for(const auto pert: {-1., -0.05, -0.01, 0.01, 0.1, 1.}) {
            auto tmpv = v2;
            v2[feature] += pert;
            auto cdist = l1dist(m, tmpv);
            assert(cdist >= cwmed || !std::fprintf(stderr, "newv: %g vs oldv %g, pert = %g", cdist, cwmed, pert));
        }
    }
}
