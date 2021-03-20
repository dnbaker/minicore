#pragma once
#ifndef MINOCORE_LSEARCHPP_H__
#define MINOCORE_LSEARCHPP_H__
#include <random>
#include <numeric>
#include <algorithm>
#include <numeric>
#include "minicore/coreset/matrix_coreset.h"
#include "minicore/util/oracle.h"
#include "minicore/util/blaze_adaptor.h"
#include "minicore/util/exception.h"
#include "libsimdsampling/simdsampling.ho.h"
#include "libsimdsampling/argminmax.ho.h"
#include "diskmat/diskmat.h"

namespace minicore {

namespace coresets {


template<typename Oracle, typename RNG, typename DistC, typename CtrsC, typename AsnT, typename WFT=double>
auto localsearchpp_rounds(const Oracle &oracle, RNG &rng, DistC &distances, CtrsC &ctrs, AsnT &asn, size_t np, size_t nrounds, const WFT *weights=nullptr, bool parallelize=true) {
    using value_type = std::decay_t<decltype(*std::begin(distances))>;
    std::uniform_real_distribution<value_type> dist;
    const unsigned k = ctrs.size();
    diskmat::PolymorphicMat<value_type> diskctrcostmat(np, k);
    auto &ctrcostmat = ~diskctrcostmat;
    if(parallelize) {
        ctrcostmat = blaze::generate(np, k, [&](auto x, auto y) {
            return oracle(x, ctrs[y]);
        });
    } else {
        for(size_t i = 0; i < np; ++i) {
            auto r = row(ctrcostmat, i);
            for(size_t j = 0; j < k; ++j) {
                r[j] = oracle(i, ctrs[j]);
            }
        }
    }
    blaze::CustomVector<value_type, blaze::unaligned, blaze::unpadded> dv(&distances[0], np);
    using CW = blaze::CustomVector<WFT, blaze::unaligned, blaze::unpadded>;
    std::unique_ptr<CW> wv;
    if(weights) {
        wv.reset(new CW((WFT *)weights, np));
    }
    dv = blz::min<blz::rowwise>(ctrcostmat);

    blz::DV<value_type> ctrcosts(k), newcosts(np);
    value_type gain;
    value_type total_gain = 0.;
    for(size_t major_round = 0; major_round < nrounds; ++major_round) {
        auto seed = rng();
        long long unsigned int sel;
        if(weights) {
            blz::DV<value_type> dv = newcosts * *wv;
            sel = reservoir_simd::sample(dv.data(), dv.size(), seed);
        } else sel = reservoir_simd::sample(distances.data(), distances.size(), seed);
        DBG_ONLY(std::fprintf(stderr, "Selected %lld with cost %g (max cost: %g)\n", sel, distances[sel], blaze::max(dv));)
        assert(sel < static_cast<long long unsigned int>(np));
        if(parallelize) {
            newcosts = blaze::generate(np, [&](auto x) {return oracle(sel, x);});
        } else {
            for(size_t i = 0; i < np; ++i) {newcosts[i] = oracle(sel, i);}
        }
        ctrcosts = 0.;
        gain = 0.;
        try {
            OMP_ONLY(_Pragma("omp parallel for reduction(+:gain)"))
            for(size_t i = 0; i < np; ++i) {
                auto row_i = row(ctrcostmat, i);
                if(const auto nc = newcosts[i], oldd = distances[i]; nc > oldd) {
                    using pt = std::pair<value_type, unsigned>;
                    pt top1 = {row_i[0], 0}, top2 = {row_i[1], 1};
                    static_assert(std::is_integral_v<decltype(top1.second)>, "");
                    static_assert(std::is_floating_point_v<decltype(top1.first)>, "");
                    if(top2.first < top1.first) std::swap(top1, top2);
                    SK_UNROLL_4
                    for(unsigned j = 2; j < k; ++j) {
                        const auto netv = row_i[j];
                        if(netv < top2.first) {
                            if(netv < top1.first) {
                                top2 = top1;
                                top1 = {netv, j};
                            } else {
                                top2 = {netv, j};
                            }
                        }
                    }
                    if(top1.first != top2.first) {
                        const auto diff = top2.first - top1.first;
                        OMP_ATOMIC
                        ctrcosts[top1.second] += diff;
                    }
                    // Now, the cost for each item is their cost - the cost of the next-lowest item
                } else {
                    gain += distances[i] - nc;
                }
            }
        } catch(const std::runtime_error &ex) {
            std::fprintf(stderr, "Warning, Caught an exception (%s), possibly nested parallel sections. Trying serial execution.\n", ex.what());
            throw;
        }
        const auto argmin = reservoir_simd::argmin(ctrcosts, /*multithread=*/true);
        const auto delta = ctrcosts[argmin] - gain;
        if(delta < 0.) {
#ifndef NDEBUG
            if(k > 25 || np > 100000)
                std::fprintf(stderr, "Swapping out %d for %lld for a gain of %g. (ctrcosts: %g. gain: %g)\n", int(ctrs[argmin]), sel, delta, ctrcosts[argmin], gain);
#endif
            ctrs[argmin] = sel;
            column(ctrcostmat, argmin, blaze::unchecked) = newcosts;
            dv = blaze::min<blaze::rowwise>(ctrcostmat);
            total_gain += delta;
        }
    }
    OMP_PFOR
    for(size_t i = 0; i < np; ++i) {
        asn[i] = reservoir_simd::argmin(row(ctrcostmat, i, blaze::unchecked), /*multithread=*/false);
    }
    return total_gain;
}

} // namespace coresets

using coresets::localsearchpp_rounds;

} // namespace minicore


#endif /* LSEARCHPP_H__ */
