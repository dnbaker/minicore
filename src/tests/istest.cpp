#include "minocore/dist/applicator.h"
#include "minocore/util/div.h"

using namespace minocore;

int main() {
    unsigned nsamp = 500;
    unsigned dim = 1000;
    unsigned nnzpr = 20;
    blaze::CompressedMatrix<double> cm(nsamp, dim);
    cm.reserve(nnzpr * nsamp);
    wy::WyRand<uint32_t, 2> rng;
    schism::Schismatic<uint32_t> div(dim);
    for(unsigned i = 0; i < nsamp; ++i) {
        for(unsigned j = nnzpr; j--;) {
            cm.set(i, div.mod(rng()), rng() % 128u);
        }
    }
    auto app = make_probdiv_applicator(cm, blz::ITAKURA_SAITO, blz::DIRICHLET);
    OMP_PFOR
    for(size_t i = 0; i < nsamp; ++i) {
        std::fprintf(stderr, "Processing row %zu\n", i + 1);
        float mnv = std::numeric_limits<float>::max();
        float mxv = -std::numeric_limits<float>::max();
        for(size_t j = 0; j < nsamp; ++j) {
            if(i != j) {
                auto v= app(i, j);
                if(v < mnv) {
                    mnv = v;
#ifndef VERBOSE_AF
                    std::fprintf(stderr, "new min: %g\n", mnv);
#endif
                } else if(v > mxv) {
                    mxv = v;
#ifndef VERBOSE_AF
                    std::fprintf(stderr, "new max: %g\n", mxv);
#endif
                }
            }
        }
        std::fprintf(stderr, "Sample %zu has its minimum distance as %g and its maximum as %g\n", i, mnv, mxv);
    }
}
