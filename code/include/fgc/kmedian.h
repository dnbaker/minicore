#pragma once
#ifndef FGC_KMEDIAN_H__
#define FGC_KMEDIAN_H__
#include "kmeans.h"
#include <algorithm>

namespace fgc {
namespace coresets {
using namespace blz;

template<typename MT, bool SO, typename VT, typename WeightType=const typename MT::ElementType *>
auto &geomedian(const blz::DenseMatrix<MT, SO> &mat, blz::DenseVector<VT, !SO> &dv, double eps=1e-8,
              const WeightType &weights=nullptr) {
    // Solve geometric median for a set of points.
    //
    using FT = typename std::decay_t<decltype(~mat)>::ElementType;
    const auto &_mat = ~mat;
    ~dv = blz::mean<blz::columnwise>(_mat);
    FT prevcost = std::numeric_limits<FT>::max();
    blz::DV<FT, !SO> costs(_mat.rows(), FT(0));
    size_t iternum = 0;
    const size_t nr = _mat.rows();
    for(;;) {
        for(size_t i = 0; i < nr; ++i) {
            auto dist = blz::l2Dist(row(_mat, i BLAZE_CHECK_DEBUG), ~dv);
            if(weights) dist *= weights[i];
            costs[i] = dist;
        }
        ++iternum;
        costs = 1. / costs;
        costs *= 1. / blaze::sum(costs);
        ~dv = costs * ~mat;
        //std::fprintf(stderr, "cost at iteration %zu: %g\n", iternum, double(blaze::sum(costs)));
        FT newcost = l2Dist(row(_mat, 0 BLAZE_CHECK_DEBUG), ~dv);
        for(size_t i = 1; i < nr; newcost += l2Dist(row(_mat, i++ BLAZE_CHECK_DEBUG), ~dv));
        if(std::abs(newcost - prevcost) <= eps) break;
        prevcost = newcost;
        std::fprintf(stderr, "Cost at iteration %zu: %g\n", iternum, newcost);
    }
    return dv;
}

template<typename FT>
static inline auto weighted_med(FT *ptr, FT *weights, size_t n, bool convex_comb=false) {
    auto cbuf(std::make_unique<std::pair<FT,FT>[]>(n));
    return weighted_med(ptr, weights, n, cbuf.get(), convex_comb);
}

} // namespace coresets
} // namespace fgc
#endif /* FGC_KMEDIAN_H__ */
