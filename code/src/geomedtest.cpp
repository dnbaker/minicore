#include "fgc/kmedian.h"

template<typename Mat, typename Vec>
auto l1dist(const Mat &m, const Vec &vec) {
    double ret = 0.;
    for(const auto r: blz::rowiterator(m)) {
        ret += blz::l1Dist(vec, r);
    }
    return ret;
}

int main() {
    blaze::DynamicMatrix<float> m(100, 2000);
    blaze::setSeed(0);
    randomize(m);
    m = pow(abs(m), -1.2);
    auto sel = rows(m, [](auto i) {return i * 2 + 1;}, 50);
    blaze::DynamicVector<float, blaze::rowVector> v, v2(2000);
    fgc::coresets::geomedian(m, v);
    std::cout << subvector(v, 0, 50) << '\n';
    fgc::coresets::l1_median(m, v2);
    std::cout << subvector(v2, 0, 50) << '\n';
    std::cout << subvector(blaze::mean<blaze::columnwise>(m), 0, 50) << '\n';
    double cwmed = l1dist(m, v2);
    std::cout << "L1 dist under geomedian: " << l1dist(m, v) << '\n';
    std::cout << "L1 dist under component-wise median: " << cwmed << '\n';
    std::cout << "L1 dist under component-wise mean: " << l1dist(m, blaze::evaluate(blaze::mean<blaze::columnwise>(m))) << '\n';
    for(unsigned feature = 0; feature < m.rows(); ++feature) {
        for(const auto pert: {-1., -0.05, -0.01, 0.01, 0.1, 1.}) {
            auto tmpv = v2;
            v2[feature] += pert;
            assert(l1dist(m, tmpv) <= cwmed);
        }
    }
}
