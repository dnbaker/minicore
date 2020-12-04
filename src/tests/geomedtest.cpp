#include "minicore/optim/kmedian.h"
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
    double ret = blz::sum(blz::abs(m - blz::expand(vec, m.rows())));
    return ret;
}
template<typename Mat, typename Vec>
auto expandl1s(const Mat &m, const Vec &vec) {
    auto ret(blz::DV<ElementType_t<Mat>>(blaze::sum<blz::rowwise>(blz::abs(m - blz::expand(vec, m.rows())))));
    return ret;
}

int main(int c, char **a) {
    unsigned dim = c < 2 ? 2000: std::atoi(a[1]);
    unsigned nr = c < 3 ? 100: std::atoi(a[2]);
    blaze::DynamicMatrix<float> m(nr, dim);
    blaze::setSeed(0);
    randomize(m);
    m = pow(abs(m), -1.2);
    blaze::DynamicVector<float, blaze::rowVector> v(dim), v2(dim), v3(dim), v4(dim);
    blz::geomedian(m, v);
    blz::DV<float, blaze::rowVector> weights = trans(blaze::generate(nr, [](auto){return 8.;}));
    blz::geomedian(m, v4, weights.data());
    auto diff = blz::l2Norm(v4 - v);
    std::fprintf(stderr, "diff between weighted and unweighted: %.12g. norms: %.12g, %.12g\n", diff, blz::l2Norm(v4), blz::l2Norm(v));
    assert(diff < 1e-4 * std::max(blz::l2Norm(v4), blz::l2Norm(v)));
    auto l1_start = t();
    minicore::coresets::l1_median(m, v2);
    auto l1_stop = t();
    //std::cout << subvector(v2, 0, nr / 2) << '\n';
    //std::cout << subvector(blaze::mean<blaze::columnwise>(m), 0, 50) << '\n';
    auto start = t();
    double cwmed = l1dist(m, v2);
    auto stop = t();
    auto manstart = t();
    double cwmed2 = reductionl1(m, v2);
    std::fprintf(stderr, "l1 diff: %0.16g/%0.16g\n", cwmed2, cwmed);
    auto manstop = t();
    std::fprintf(stderr, "Manual l1 dist time: %zu/%g. reduction-based: %zu/%g\n", size_t((stop - start).count() / 1000), cwmed, size_t((manstop - manstart).count() / 1000), cwmed2);
    start = t();
    auto l1distances = l1dists(m, v2);
    stop = t();
    manstart = t();
    auto l1distances2 = expandl1s(m, v2);
    manstop = t();
    std::fprintf(stderr, "Manual l1 distances time: %zu/%g. reduction-based: %zu/%g\n", size_t((stop - start).count() / 1000), cwmed, size_t((manstop - manstart).count() / 1000), cwmed2);
    auto l1_approx_start = t();
    minicore::coresets::l1_median(m, v3, static_cast<const float *>(nullptr));
    auto l1_approx_stop = t();
    std::fprintf(stderr, "Time to compute exact l1 median: %gms. Approx: %gms.\n", (l1_stop - l1_start).count() * 1.e-6, (l1_approx_stop - l1_approx_start).count() * 1.e-6);
    std::cout << "L1 dist under geomedian: " << l1dist(m, v) << '\n';
    std::cout << "L1 dist under component-wise median: " << cwmed << '\n';
    std::cout << "L1 dist under approx component-wise median: " << reductionl1(m, v3) << '\n';
    std::cout << "L1 dist under component-wise mean: " << l1dist(m, blaze::evaluate(blaze::mean<blaze::columnwise>(m))) << '\n';
    for(unsigned feature = 0; feature < m.columns(); ++feature) {
        for(const auto pert: {-1., -0.05, -0.01, -0.001, 0.001, 0.00001, 0.01, 0.1, 1.}) {
            auto tmpv = v2;
            v2[feature] += pert;
            if(l1dist(m, tmpv) - cwmed2 <= 0)
                std::fprintf(stderr, "feature %d. newv: %0.20g vs oldv %0.20g (diff %0.12g), pert = %g\n", feature, l1dist(m, tmpv), cwmed, l1dist(m, tmpv) - cwmed, pert);
        }
    }
}
