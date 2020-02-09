#include "fgc/diskmat.h"
#include "fgc/distance.h"

using namespace fgc;
using namespace blz;

int main() {
    const char *fn = "./zomg.dat";
    {
        DiskMat<double> dm(200, 1000, fn);
        dm.delete_file_ = true;
        assert(::access(fn, F_OK) != -1);
    }
    assert(::access(fn, F_OK) == -1);
    DiskMat<double> dm(200, 1000, fn);
    dm.delete_file_ = true;
    ~dm = 0;
    assert(dm(1, 4) == 0.);
    assert(blaze::sum(~dm) == 0);
    randomize(~dm);
    ~dm = abs(~dm + 1e-15);
    auto r1 = row(dm, 1), r0 = row(dm, 0);
    randomize(r1);
    randomize(r0);
    r1[0] = 0.;
    r0[0] = 0.;
    r1[4] = 0.;
    r1 *= r1;
    r0 *= r0;
    r1 /= l2Norm(r1);
    r0 /= l2Norm(r0);
    for(size_t i = 0; i < 200u; ++i)
        row(dm, i) /= l2Norm(row(dm, i));
    blaze::DynamicMatrix<double> cpy(~dm);
    MultinomialJSDApplicator<decltype(cpy)> jsd(cpy);
    blaze::CompressedVector<double> c1(r0.size()), c0(r0.size());
    c1.reserve(r1.size() - 2);
    c0.reserve(r0.size() - 1);
    for(size_t i = 0; i < r0.size(); ++i) {
        if(r0[i])
            c0.append(i, r0[i]);
        if(r1[i])
            c1.append(i, r1[i]);
    }
    blaze::CompressedVector<double> c3(4000, 200), po(4000, 200), c3po(4000, 100);
    for(size_t i = 0; i < 100; ++i) {
        auto ind = std::rand() % 4000;
        c3[ind] = 1. / 200;
        po[ind] = 1. / 200;
        c3po[ind] = 1. / 100;
        c3[std::rand() % 4000] = 1. / 200;
        po[std::rand() % 4000] = 1. / 200;
    }
    blaze::DynamicVector<double> dc3 = c3, dpo = po, dc3po = c3po;
    std::fprintf(stderr, "multinomial jsd: %f\n", distance::multinomial_jsd(r1, r0));
    std::fprintf(stderr, "multinomial jsd: %f\n", distance::multinomial_jsd(c1, c0));
    std::fprintf(stderr, "multinomial jsd from applicator: %f\n", jsd(0, 1));
    auto cdpo_jsd = distance::multinomial_jsd(c3, po);
    auto c32  = distance::multinomial_jsd(c3, c3po);
    std::fprintf(stderr, "Sparse,random vectors sharing a subset: %g\n", cdpo_jsd);
    std::fprintf(stderr, "Sparse,random vectors sharing a subset: %g\n", c32);
    std::fprintf(stderr, "Sparse,random vectors sharing a subset: %g\n", distance::multinomial_jsd(c3, c3));
    std::fprintf(stderr, "Dense versions of random vector and itself: %g\n", distance::multinomial_jsd(dc3, dc3));
    std::fprintf(stderr, "Dense versions of random vectors sharing a subset: %g\n", distance::multinomial_jsd(dc3, dpo));
    std::fprintf(stderr, "Dense versions of random vectors sharing a subset, no filter: %g\n", distance::multinomial_jsd(dc3, dpo, distance::FilterNans<false>()));
}
