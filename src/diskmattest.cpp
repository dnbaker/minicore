#include "minocore/dist.h"
#include "minocore/util/diskmat.h"

using namespace minocore;
using namespace blz;

int main() {
    std::srand(0);
    unsigned nrows = 50, ncol = 200;
    const char *fn = "./zomg.dat";
    {
        DiskMat<double> dm(nrows, ncol, fn);
        dm.delete_file_ = true;
        assert(::access(fn, F_OK) != -1);
    }
    assert(::access(fn, F_OK) == -1);
    DiskMat<double> dm(nrows, ncol, fn);
    dm.delete_file_ = true;
    ~dm = 0;
    assert(dm(1, 4) == 0.);
    assert(blaze::sum(~dm) == 0);
    ~dm = blaze::generate((~dm).rows(), (~dm).columns(), [](auto x, auto y) {
        wy::WyRand<uint64_t, 2> rng((uint64_t(x) << 32) | y);
        std::uniform_real_distribution<float> urd;
        return urd(rng);
     });
    auto r1 = row(dm, 1), r0 = row(dm, 0);
    r1[0] = 0.;
    r0[0] = 0.;
    r1[4] = 0.;
    r1 *= r1;
    r0 *= r0;
    r1 /= l2Norm(r1);
    r0 /= l2Norm(r0);
    for(size_t i = 0; i < nrows; ++i)
        row(dm, i) /= l2Norm(row(dm, i));
    blaze::DynamicMatrix<double> cpy(~dm);
    blaze::DynamicMatrix<double> cpy2(~dm);
    for(unsigned i = 0; i < cpy.rows() * cpy.columns() * 9 / 10; ++i) {
        auto rn = std::rand();
        auto rownum = rn % cpy.rows();
        auto cnum = (rn / cpy.rows()) % cpy.columns();
        cpy2(rownum, cnum) = 0.;
    }
    blaze::CompressedMatrix<double> sparse_cpy(cpy2);
    {
        blaze::DynamicMatrix<double> tmp; std::swap(tmp, cpy2);
    }
    blaze::CompressedVector<double> c1(r0.size()), c0(r0.size());
    c1.reserve(r1.size() - 2);
    c0.reserve(r0.size() - 1);
    for(size_t i = 0; i < r0.size(); ++i) {
        if(r0[i])
            c0.append(i, r0[i]);
        if(r1[i])
            c1.append(i, r1[i]);
    }
    std::cout << r1;
    std::cout << r0;
    std::fprintf(stderr, "Wasserstein distance between rows 1 and 2: %g\n", distance::p_wasserstein(r1, r0));
#if 0
    std::fprintf(stderr, "multinomial jsd: %f\n", distance::multinomial_jsd(r1, r0));
    std::fprintf(stderr, "multinomial jsd: %f\n", distance::multinomial_jsd(c1, c0));
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
    std::fprintf(stderr, "multinomial jsd from applicator: %f\n", jsd(0, 1));
    std::fprintf(stderr, "sparse jsd from sparse applicator: %f\n", sparse_jsd(0, 1));
    auto cdpo_jsd = distance::multinomial_jsd(c3, po);
    auto c32  = distance::multinomial_jsd(c3, c3po);
    auto distmat = jsd.make_distance_matrix();
    std::fprintf(stderr, "Sparse,random vectors sharing a subset: %g\n", cdpo_jsd);
    std::fprintf(stderr, "Sparse,random vectors sharing a subset: %g\n", c32);
    std::fprintf(stderr, "Sparse,random vectors sharing a subset: %g\n", distance::multinomial_jsd(c3, c3));
    std::fprintf(stderr, "Dense versions of random vector and itself: %g\n", distance::multinomial_jsd(dc3, dc3));
    std::fprintf(stderr, "Dense versions of random vectors sharing a subset: %g\n", distance::multinomial_jsd(dc3, dpo));
    //std::fprintf(stderr, "Dense versions of random vectors sharing a subset, no filter: %g\n", distance::multinomial_jsd(dc3, dpo, distance::FilterNans<false>()));
#endif
}
