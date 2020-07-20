#pragma once
#ifndef MINOCORE_LSEARCHPP_H__
#define MINOCORE_LSEARCHPP_H__
#include <random>
#include <numeric>
#include <algorithm>
#include <numeric>
#include "minocore/coreset/matrix_coreset.h"
#include "minocore/util/oracle.h"
#include "minocore/util/blaze_adaptor.h"
#include "minocore/util/exception.h"

namespace minocore {

namespace coresets {


template<typename Oracle, typename RNG, typename DistC, typename CtrsC, typename AsnT, typename WFT=double>
auto localsearchpp_rounds(const Oracle &oracle, RNG &rng, DistC &distances, DistC &cdf, CtrsC &ctrs, AsnT &asn, size_t np, size_t nrounds, const WFT *weights=nullptr) {
    using value_type = std::decay_t<decltype(*std::begin(distances))>;
    //blz::DV<value_type> newdists = blaze::CustomVector<value_type, blaze::unaligned, blaze::unpadded>(distances.data(), distances.size());
    std::uniform_real_distribution<value_type> dist;
    auto make_sel = [&]() -> std::ptrdiff_t {return std::lower_bound(cdf.begin(), cdf.end(), dist(rng) * cdf.back()) - cdf.begin();};
    const unsigned k = ctrs.size();
    blz::DM<value_type> ctrcostmat = blaze::generate(np, k, [&](auto x, auto y) {
        return oracle(x, ctrs[y]);
    });
    blaze::CustomVector<value_type, blaze::unaligned, blaze::unpadded> dv(&distances[0], np);
    dv = blz::min<blz::rowwise>(ctrcostmat);
#ifndef NDEBUG
    auto ccost = blz::sum(dv);
    const auto ocost = ccost;
    std::fprintf(stderr, "Cost before lsearch++: %g\n", ccost);
#endif

    blz::DV<value_type> ctrcosts(k), newcosts(np);
    value_type gain = 0.;
    value_type total_gain = 0.;
    for(size_t major_round = 0; major_round < nrounds; ++major_round) {
        const auto sel = make_sel();
#ifndef NDEBUG
        std::fprintf(stderr, "Selected %td with cost %g (max cost: %g)\n", sel, distances[sel], blaze::max(dv));
#endif
        assert(sel < static_cast<std::ptrdiff_t>(np));
        newcosts = blaze::generate(np, [&](auto x) {return oracle(sel, x);});
        ctrcosts = 0.;
        for(size_t i = 0; i < np; ++i) {
            auto row_i = row(ctrcostmat, i);
            if(const auto nc = newcosts[i], oldd = distances[i]; nc > oldd) {
                using pt = std::pair<value_type, unsigned>;
                pt top1 = {row_i[0], 0}, top2 = {row_i[1], 1};
                static_assert(std::is_integral_v<decltype(top1.second)>, "");
                static_assert(std::is_floating_point_v<decltype(top1.first)>, "");
                if(top2.first < top1.first) std::swap(top1, top2);
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
#ifndef NDEBUG
                std::vector<pt> pairs;
                for(unsigned i = 0; i < k; ++i) pairs.push_back({row_i[i], i});
                std::sort(pairs.begin(), pairs.end());
                //for(unsigned i = 0; i < k; ++i) std::fprintf(stderr, "sorted entry %u is %g/%u\n", i, pairs[i].first, pairs[i].second);
                //std::fprintf(stderr, "Top 2 items: %g/%u, %g/%u\n", top1.first, top1.second, top2.first, top2.second);
                assert(top1.first == pairs[0].first);
                assert(top2.first == pairs[1].first);
#endif
                if(top1.first != top2.first)
                    ctrcosts[top1.second] += top2.first - top1.first;
                // Now, the cost for each item is their cost -
                // the cost of the next-lowest item
            } else {
                gain += distances[i] - nc;
            }
        }
        for(unsigned i = 0; i < k; ++i) {
            std::fprintf(stderr, "Center %u has %u for id and %g for costs\n", i, ctrs[i], ctrcosts[i]);
        }
        const auto argmin = std::min_element(ctrcosts.begin(), ctrcosts.end()) - ctrcosts.begin();
        const auto delta = ctrcosts[argmin] - gain;
        if(delta < 0.) {
            std::fprintf(stderr, "Swapping out %d for %zd for a gain of %g. (ctrcosts: %g. gain: %g)\n", ctrs[argmin], sel, delta, ctrcosts[argmin], gain);
            ctrs[argmin] = sel;
            std::fprintf(stderr, "newcosts size: %zu. ctrcost dims: %zu/%zu\n", newcosts.size(), ctrcostmat.rows(), ctrcostmat.columns());
            column(ctrcostmat, argmin, blaze::unchecked) = newcosts;
            dv = blaze::min<blaze::rowwise>(ctrcostmat);
#ifndef NDEBUG
            auto nextccost = blz::sum(dv);
            std::fprintf(stderr, "Cost before lsearch++: %g. After: %g\n", ccost, nextccost);
            ccost = nextccost;
#endif
            total_gain += delta;
            if(major_round + 1 != nrounds) {
                if(weights) ::std::partial_sum(distances.begin(), distances.end(), cdf.begin(), [weights,ds=&distances[0]](auto x, const auto &y) {
                    return x + y * weights[&y - ds];
                });
                else ::std::partial_sum(distances.begin(), distances.end(), cdf.begin());
            }
        }
    }
    for(size_t i = 0; i < np; ++i) {
        auto r = row(ctrcostmat, i, blaze::unchecked);
        asn[i] = std::min_element(r.begin(), r.end()) - r.begin();
    }
#ifndef NDEBUG
    std::fprintf(stderr, "Cost before %zu rounds of lsearch++: %g. After: %g\n", nrounds, ocost, ccost);
#endif
    return total_gain;
}

} // namespace coresets

using coresets::localsearchpp_rounds;

} // namespace minocore


#endif /* LSEARCHPP_H__ */
