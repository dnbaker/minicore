#include "minocore/utility.h"
#include "minocore/dist.h"

#ifndef DENSESUB
#define SUBMAT CompressedMatrix
#else
#define SUBMAT DynamicMatrix
#endif

#define FLOAT_TYPE double
int main(int ac, char **av) {
    auto sm = minocore::csc2sparse<FLOAT_TYPE>("");
    unsigned nr = ac > 1 ? std::atoi(av[1]): 100;
    unsigned mc = ac > 2 ? std::atoi(av[2]): 50;
    uint32_t seed = ac > 4 ? std::atoi(av[4]): 0;
    std::srand(seed);
    std::vector<unsigned> indices;
    while(indices.size() < nr && indices.size() < sm.rows()) {
        auto ind = std::rand() % sm.rows();
        if(blaze::nonZeros(row(sm, ind)) > mc && std::find(indices.begin(), indices.end(), ind) == indices.end())
            indices.push_back(ind);
    }
    if(indices.size() > nr) nr = indices.size();
    blaze::SUBMAT<FLOAT_TYPE> subm(rows(sm, indices.data(), indices.size()));
#if DENSESUB
    double inc = ac > 3 ? std::atof(av[3]): 1.;
    subm += inc;
    //subm = (subm % subm % subm) + inc;
#endif
    auto app = minocore::jsd::make_probdiv_applicator(subm, blz::LLR);
    blaze::DynamicMatrix<FLOAT_TYPE> distmat(subm.rows(), subm.rows());
    auto t = std::chrono::high_resolution_clock::now();
    app.set_distance_matrix(distmat);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "Distmat took %g ms\n", double((t2 - t).count()) / 1e6);
    size_t maxi = -1, maxj = -1, maxk = -1;
    std::set<unsigned> tmpind;
    while(int(tmpind.size()) < std::min(50, int(nr))) tmpind.insert(std::rand() % subm.rows());
    std::vector<unsigned> iv(tmpind.begin(), tmpind.end());
    FLOAT_TYPE mxv = 0., mxsq = 0.;
    for(size_t i = 0; i < nr; ++i) {
        for(size_t j = i + 1; j < nr; ++j) {
            for(const auto k: iv) {
                if(k == j || k == i) continue;
                auto xy = distmat(i, j);
                if(xy == 0) continue;
                auto xz = distmat(i, k);
                if(xz == 0) continue;
                auto yz = distmat(j, k);
                if(yz == 0) continue;
                auto xys = std::sqrt(xy),  yzs = std::sqrt(yz), xzs = std::sqrt(xz);
                FLOAT_TYPE alpha, asq;
                if(xy > xz) {
                    if(xy > yz) {
                        alpha = xy / (xz + yz), asq = xys / (xzs + yzs);
                    } else {
                        alpha = yz / (xz + xy), asq = yzs / (xys + xzs);
                    }
                } else if(yz > xz) {
                    alpha = yz / (xz + xy), asq = yzs / (xys + xzs);
                } else alpha = xz / (yz + xy), asq = xzs / (yzs + xys);
                if(std::isinf(alpha)) {
                    std::fprintf(stderr, "%g/%g/%g provided infinite alpha, skipping\n", xy, xz, yz);
                    continue;
                }
                xz = std::sqrt(xz);
                xy = std::sqrt(xy);
                yz = std::sqrt(yz);
                if(alpha > mxv) {
                    std::fprintf(stderr, "new alpha %g > old %g at %zu/%zu/%u\n", alpha, mxv, i, j, k);
                    mxv = alpha;
                    maxi = i; maxj = j; maxk = k;
                }
                mxsq = std::max(mxsq, asq);
            }
        }
    }
    std::fprintf(stderr, "biggest alpha %g at %zu/%zu/%zu. alphas: %g\n", mxv, maxi, maxj, maxk, mxsq);
}
