#pragma once
#ifndef FGC_KMEDIAN_H__
#define FGC_KMEDIAN_H__
#include "kmeans.h"
#include <algorithm>
#include "pdqsort/pdqsort.h"

namespace fgc {
namespace coresets {

// Median of a weighted set
template<typename FT>
static inline auto weighted_med(FT *ptr, FT *weights, size_t n, std::pair<FT,FT> *cbuf, bool convex_comb=false) {
    //auto csn = nonstd::span<std::pair<FT,FT>>(cbuf.get(), n);
    for(size_t i = 0; i < n; ++i) {
        cbuf[i] = {ptr[i], weights[i]};
    }
    pdqsort(cbuf, cbuf + n, [](std::pair<FT, FT> x, std::pair<FT, FT> y) {return x.first < y.first;});
    auto wsum = 0.;
    for(auto it = &cbuf[0], eit = &cbuf[n]; it < eit; ++it) {
        auto &i = *it;
        wsum += i.second;
        i.second = wsum;
    }
    FT halfwsum = wsum / 2.;
    assert(std::is_sorted(&cbuf[0], &cbuf[n], [](auto x, auto y) {return x.second < y.second;}));
    auto it = std::lower_bound(&cbuf[0], &cbuf[n], halfwsum, [](std::pair<FT, FT> x, FT y) {return x.second < y;});
    if(convex_comb) {
        auto d1 = std::abs(it->second - halfwsum), d2 = std::abs((it + 1)->second - halfwsum);
        std::fprintf(stderr, "diffs: %f, %f. halfwsum: %f\n", d1, d2, halfwsum);
        return (d1 * (it + 1)->first + d2 * it->first) / (d1 + d2);
    }
    return it->first;
}

template<typename FT>
static inline auto weighted_med(FT *ptr, FT *weights, size_t n, bool convex_comb=false) {
    auto cbuf(std::make_unique<std::pair<FT,FT>[]>(n));
    return weighted_med(ptr, weights, n, cbuf.get(), convex_comb);
}

template<typename It, typename Cmp=std::less<>>
INLINE void sort(It beg, It end, Cmp cmp=Cmp()) {
#ifdef PDQSORT_H
    pdqsort(beg, end, cmp);
#else
    std::sort(beg, end, cmp);
}
#endif
#if 0
template<typename IT, typename MatrixType, typename CMatrixType=MatrixType, typename WFT=double>
                           CMatrixType &centers, MatrixType &data,
                           const WFT *weights=nullptr)
{
    auto getw = [weights](size_t ind) {return weights ? weights[ind]: WFT(1.);};
    // 1. Calculate centers from assignments
    using FT = typename ContainedTypeFromIterator<Iter>;
    for(size_t i = 0; i < data.rows(); ++i) {
        auto asn = assignments[i];
        auto r = row(data, i);
        auto &vvec = vecs[asn];
        for(size_t j = 0; j < centers.columns(); ++j) {
            vvec[j].push_back(r[j]);
        }
    }
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0, e = centers.rows() * centers.columns(); i < e; ++i) {
        auto centernum = i / centers.columns(), cind = i % centers.columns();
        auto &v = vecs[centernum][cind];
        sort(v.begin(), v.end());
        centers(centernum, cind) = v.size () & 1 ? v[v.size() / 2]: FT((v[v.size() / 2] + v[(v.size() - 1) / 2]) * .5);
    }
    // 2. Calculate assignments from centers
}
template<typename IT, typename MatrixType, typename CMatrixType=MatrixType, typename WFT=double>
void kmedian_em_loop(std::vector<IT> &assignments, std::vector<WFT> &counts,
                     CMatrixType &centers, MatrixType &data,
                     double tolerance=.001, size_t maxiter=-1,
                     const WFT *weights=nullptr)
{
    if(tolerance < 0.) throw 1;
    double oldloss = kmeans_em_iteration(assignments, counts, centers, data, weights);
    size_t iternum = 0;
    for(;;) {
        double newloss = kmeans_em_iteration(assignments, counts, centers, data, weights);
        if(std::abs(oldloss - newloss) / oldloss < tolerance || iternum++ == maxiter) return;
        oldloss = newloss;
        std::fprintf(stderr, "loss at %zu: %g\n", iternum, oldloss);
    }
}
#endif
} // namespace coresets
} // namespace fgc
#endif /* FGC_KMEDIAN_H__ */
