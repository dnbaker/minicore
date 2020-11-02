#undef NDEBUG
#include "minicore/dist/applicator.h"
#include <cassert>

#define FT double
using namespace minicore;
using namespace minicore::distance::detail;


int main() {
    blz::CompressedVector<FT> cv1{{10., 0., 4., 6., 0., 13}};
    blz::CompressedVector<FT> cv2 = cv1 * 2;
    blz::CompressedVector<FT> cv3 = cv1;
    blz::CompressedVector<FT> cv4 = {0., 0., 0., 1.,};
    blz::CompressedVector<FT> cv5 = {1., 1., 1., 1.,};
    blz::DV<FT> data({10, 4, 6, 13});
    blz::DV<uint64_t> indices{0, 2, 3, 5};
    util::CSparseVector<FT, uint64_t> csv(data.data(), indices.data(), 4, 10);
    for(const auto &pair: csv) {
        std::fprintf(stderr, "%zu/%g\n", pair.index(), pair.value());
        assert(cv1[pair.index()] == pair.value());
    }
    const size_t nd = cv1.size();
    const std::vector<blz::CompressedVector<FT> *> ptrs{&cv1, &cv2, &cv3, &cv4, &cv5};
    for(auto x: ptrs) {
        x->resize(nd);
    }
    blz::DV<FT> prior(1);
    FT psum = prior[0] * cv1.size();
    prior[0] = 1.;
    FT s1 = sum(cv1), s2 = sum(cv2), s3 = sum(cv3);
    int anyfail = 0.;
    for(const auto msr: minicore::distance::detail::USABLE_MEASURES) {
        if(msr == distance::SYMMETRIC_ITAKURA_SAITO) {
            break;
        }
        for(const auto pval: {0., 1e-5, 1., 50.}) {
            std::fprintf(stderr, "Msr %d/%s with prior %g: %s. \n", (int)msr, prob2str(msr), pval, prob2desc(msr));
            psum = pval * nd;
            prior[0] = pval;
            try {
            auto v = cmp::msr_with_prior(msr, cv1, cv1, prior, psum, s1, s1);
            if(msr == minicore::distance::COSINE_SIMILARITY) {
                if(v != 1.) assert(v == 1.);
                continue;
            }
            if(v != 0) {
                std::fprintf(stderr, "FAILURE (both blaze): %g != 0.\n", v);
                if(v > 5e-9) {
                    assert(v == 0.);
                }
                ++anyfail;
            }
            if(std::isnan(v)) {
                std::fprintf(stderr, "ISNAN: %g\n", v);
                assert(std::isnan(v));
                ++anyfail;
            }
            if(auto v = cmp::msr_with_prior(msr, cv1, cv4, prior, psum, sum(cv1), sum(cv4)); v < 0) {
                assert(v >= 0);
            }
            } catch(const exception::TODOError &ex) {
                // don't care
            }
#if 0
            std::cerr << cv1;
            std::cerr << csv;
            auto v2 = cmp::msr_with_prior(msr, cv1, csv, prior, psum, s1, s1);
            if(v2 != 0) {
                std::fprintf(stderr, "FAILURE (only one blaze): %g != 0.\n", v2);
                assert(v2 == 0.);
                ++anyfail;
            }
            if(std::isnan(v2)) {
                std::fprintf(stderr, "ISNAN: %g\n", v2);
                assert(std::isnan(v2));
                ++anyfail;
            }
#endif
        }
    }
    std::fprintf(stderr, "%d failures\n", anyfail);
}
