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

namespace minocore {

namespace coresets {


template<typename Oracle, typename RNG, typename DistC, typename CtrsC, typename AsnT, typename WFT=double>
auto localsearchpp_rounds(const Oracle &oracle, RNG &rng, DistC &distances, DistC &cdf, CtrsC &ctrs, AsnT &asn, size_t np, size_t nrounds, const WFT *weights=nullptr) {
    using value_type = std::decay_t<decltype(*std::begin(distances))>;
    //blz::DV<value_type> newdists = blaze::CustomVector<value_type, blaze::unaligned, blaze::unpadded>(distances.data(), distances.size());
    std::uniform_real_distribution<value_type> dist;
    auto make_sel = [&]() -> std::ptrdiff_t {return std::lower_bound(cdf.begin(), cdf.end(), dist(rng) * cdf.back()) - cdf.begin();};
    blz::DM<value_type> ctrcostmat = blaze::generate(np, ctrs.size(), [&](auto x, auto y) {
        return oracle(x, ctrs[y]);
    });
    blaze::CustomVector<value_type, blaze::unaligned, blaze::unpadded> dv(&distances[0], np);
    dv = blz::min<blz::rowwise>(ctrcostmat);

    blz::DV<value_type> ctrcosts(ctrs.size()), newcosts(np);
    value_type gain = 0.;
    value_type total_gain = 0.;
    for(size_t major_round = 0; major_round < nrounds; ++major_round) {
        const auto sel = make_sel();
        std::fprintf(stderr, "Selected %td with cost %g (max cost: %g)\n", sel, distances[sel], blaze::max(dv));
        assert(sel < static_cast<std::ptrdiff_t>(np));
        newcosts = blaze::generate(np, [&](auto x) {return oracle(sel, x);});
        ctrcosts = 0.;
        for(size_t i = 0; i < np; ++i) {
            if(const auto nc = newcosts[i], oldd = distances[i]; nc >= oldd) {
                throw NotImplementedError("Proper calculation of the cost of removal of nodes");
#if 0
                std::fprintf(stderr, "newcost %g, oldcost %g. min cost in matrix: %g\n", nc, distances[i],  blz::min(row(ctrcostmat, i)));
                assert(std::abs(distances[i] - blz::min(row(ctrcostmat, i))) < 1e-6); // the distance should be the minimum cost
                ctrcosts += trans(row(ctrcostmat, i) - nc);
#endif
            } else {
                gain += distances[i] - nc;
            }
        }
        for(unsigned i = 0; i < ctrs.size(); ++i) {
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
            total_gain += delta;
            if(weights) ::std::partial_sum(distances.begin(), distances.end(), cdf.begin(), [weights,ds=&distances[0]](auto x, const auto &y) {
                return x + y * weights[&y - ds];
            });
            else ::std::partial_sum(distances.begin(), distances.end(), cdf.begin());
        }
    }
    for(size_t i = 0; i < np; ++i) {
        auto r = row(ctrcostmat, i, blaze::unchecked);
        asn[i] = std::min_element(r.begin(), r.end()) - r.begin();
    }
    return total_gain;
}

#if 0
template<typename Oracle, typename RNG, typename DistC, typename CtrsC, typename AsnC, typename WFT=double>
void localsearchpppp_rounds(const Oracle &oracle, RNG &rng, DistC &distances, DistC &cdf, CtrsC &ctrs, typename AsnC, size_t np, int k, size_t nrounds, int batchsize=3, const WFT *weights=nullptr) {
    using value_type = std::decay_t<decltype(*std::begin(distances))>;
    //blz::DV<value_type> newdists = blaze::CustomVector<value_type, blaze::unaligned, blaze::unpadded>(distances.data(), distances.size());
    std::uniform_real_distribution<value_type> dist;
    auto make_sel = [&]() -> std::ptrdiff_t {return std::lower_bound(cdf.begin(), cdf.end(), dist(rng) * cdf.back()) - cdf.begin();};
    blz::DM<value_type> ctrcostmat = blaze::generate(np, ctrs.size(), [&](auto x, auto y) {
        return oracle(ctrs[x], y);
    });

    blz::DV<value_type> ctrcosts(ctrs.size());
    blaze::DM<value_type> newcosts(batchsize, np), ncv(np);
    value_type gain = 0.;
    value_type total_gain = 0.;
    std::vector<unsigned> idx(batchsize);
    for(size_t major_round = 0; major_round < nrounds; ++major_round) {
        for(auto &i: idx) i = make_sel();
        newcosts = blaze::generate(batchsize, np, [&](auto x, auto y) {return oracle(idx[x], y);});
        ncv = blaze::min<rowwise>(newcosts);
        ctrcosts = 0.;
        for(size_t i = 0; i < np; ++i) {
            if(ncv[i] >= distances[i]) {
                ctrcosts += row(ctrcostmat, i) - newcv;
            } else {
                gain += distances[i] - newcosts[i];
            }
        }
        auto argmin = std::min_element(ctrcosts.begin(), ctrcosts.end(),) - ctrcosts.begin();
        auto delta = ctrcosts[i] - gain;
        if(delta < 0.) {
            std::fprintf(stderr, "Swapping out %zd for %zd for a gain of %g\n", ctrs[argmin], sel, delta);
            ctrs[argmin] == sel;
            row(ctrcostmat, argmin, blaze::unchecked) = newcosts;
            blaze::CustomVector<value_type, blaze::unaligned, blaze::unpadded>(&distances[0], np) =
                blaze::min<rowwise>(blaze::trans(ctrcostmat));
            total_gain += delta;
        }
    }
    return total_gain;
}
#endif

}

}


#endif /* LSEARCHPP_H__ */
