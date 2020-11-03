#undef NDEBUG
#include "minicore/dist/applicator.h"
#include <cassert>

#define FT double
using namespace minicore;
using namespace minicore::distance::detail;


int main() {
    blz::CompressedVector<FT> cv1{{10., 0., 4., 6.}};
    blz::CompressedVector<FT> cv2 = cv1 * 2;
    blz::CompressedVector<FT> cv3 = cv1;
    blz::CompressedVector<FT> cv4 = {0., 0., 0., 1.,};
    blz::CompressedVector<FT> cv5 = {1., 1., 1., 1.,};
    blz::DV<FT> data({10, 4, 6});
    blz::DV<uint64_t> indices{0, 2, 3};
    util::CSparseVector<FT, uint64_t> csv(data.data(), indices.data(), 3, 10);
    for(const auto &pair: csv) {
        std::fprintf(stderr, "%zu/%g\n", pair.index(), pair.value());
        assert(cv1[pair.index()] == pair.value());
    }
    const std::vector<blz::CompressedVector<FT> *> ptrs{&cv1, &cv2, &cv3, &cv4, &cv5};
    for(auto x: ptrs) {
        x->resize(10);
    }
    blz::DV<FT> prior(1);
    FT psum = prior[0] * cv1.size();
    prior[0] = 1.;
    FT s1 = sum(cv1);
    int anyfail = 0.;
    auto checkv = [&](auto v, auto dis) {
        if(v != 0) {
            std::fprintf(stderr, "FAILURE (both blaze): %g != 0.\n", v);
            if(v > 1e-8) {
                assert(v == 0. || dis == minicore::distance::BHATTACHARYYA_METRIC);
            }
            ++anyfail;
        }
    };
    for(const auto msr: minicore::distance::detail::USABLE_MEASURES) {
        if(msr == distance::SYMMETRIC_ITAKURA_SAITO || msr == minicore::distance::COSINE_SIMILARITY) {
            continue;
        }
        for(const auto pval: {0., 1e-5, 1., 50.}) {
            std::fprintf(stderr, "Msr %d/%s with prior %g: %s. \n", (int)msr, prob2str(msr), pval, prob2desc(msr));
            psum = pval * cv1.size();
            prior[0] = pval;
            try {
            auto v = cmp::msr_with_prior(msr, csv, csv, prior, psum, s1, s1);
            assert(v == cmp::msr_with_prior(msr, cv1, csv, prior, psum, s1, s1));
            if(pval == 0. && std::isnan(v)) v = 0.;
            checkv(v, msr);
            auto v2 = cmp::msr_with_prior(msr, csv, cv4, prior, psum, s1, sum(cv4));
            assert(!std::isnan(v));
            assert(!std::isnan(v2) || pval == 0. || sum(cv4) == 0.);
            if(!std::isnan(v2)) assert(v2 > 0. || !std::fprintf(stderr, "v2: %g\n", v2));
            } catch(const exception::TODOError &ex) {
                // don't care
            }
        }
    }
    std::fprintf(stderr, "%d failures\n", anyfail);
}
