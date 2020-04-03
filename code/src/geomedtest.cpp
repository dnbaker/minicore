#include "fgc/kmedian.h"

int main() {
    blaze::DynamicMatrix<float> m(100, 2000);
    blaze::setSeed(0);
    randomize(m);
    m = pow(abs(m), -2);
    auto sel = rows(m, [](auto i) {return i * 2 + 1;}, 50);
    blaze::DynamicVector<float, blaze::rowVector> v;
    fgc::coresets::geomedian(m, v);
}
