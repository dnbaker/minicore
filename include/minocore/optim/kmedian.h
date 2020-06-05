#pragma once
#ifndef FGC_KMEDIAN_H__
#define FGC_KMEDIAN_H__
#include "minocore/optim/kmeans.h"
#include <algorithm>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/p_square_cumul_dist.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/sum.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/weighted_median.hpp>

namespace minocore {
namespace coresets {
using namespace blz;
using namespace boost::accumulators;

template<typename MT, bool SO, typename VT, typename WeightType=const typename MT::ElementType *>
auto &geomedian(const blz::DenseMatrix<MT, SO> &mat, blz::DenseVector<VT, !SO> &dv, double eps=1e-8,
              const WeightType &weights=nullptr) {
    // Solve geometric median for a set of points.
    //
    using FT = typename std::decay_t<decltype(~mat)>::ElementType;
    const auto &_mat = ~mat;
    ~dv = blz::mean<blz::columnwise>(_mat);
    FT prevcost = std::numeric_limits<FT>::max();
    blaze::DynamicVector<FT, !SO> costs(_mat.rows(), FT(0));
    size_t iternum = 0;
    const size_t nr = _mat.rows();
    assert((~dv).size() == (~mat).columns());
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
        FT newcost = l1Dist(row(_mat, 0 BLAZE_CHECK_DEBUG), ~dv);
        for(size_t i = 1; i < nr; newcost += l1Dist(row(_mat, i++ BLAZE_CHECK_DEBUG), ~dv));
        if(std::abs(newcost - prevcost) <= eps) break;
        prevcost = newcost;
    }
    return dv;
}

template<typename MT, bool SO, typename VT, bool TF>
void l1_unweighted_median(const blz::Matrix<MT, SO> &data, blz::DenseVector<VT, TF> &ret) {
    assert((~ret).size() == (~data).columns());
    auto &rr(~ret);
    const auto &dr(~data);
    const bool odd = dr.rows() % 2;
    const size_t hlf = dr.rows() / 2;
    for(size_t i = 0; i < dr.columns(); ++i) {
        blaze::DynamicVector<ElementType_t<MT>, blaze::columnVector> tmpind = column(data, i); // Should do fast copying.
        shared::sort(tmpind.begin(), tmpind.end());
        rr[i] = odd ? tmpind[hlf]: ElementType_t<MT>(.5) * (tmpind[hlf] + tmpind[hlf + 1]);
    }
}

template<typename MT, bool SO, typename VT, bool TF, typename Rows>
void l1_unweighted_median(const blz::Matrix<MT, SO> &_data, const Rows &rs, blz::DenseVector<VT, TF> &ret) {
    assert((~ret).size() == (~_data).columns());
    auto &rr(~ret);
    const auto &dr(~_data);
    const bool odd = rs.size() % 2;
    const size_t hlf = rs.size() / 2;
    const size_t nc = dr.columns();
    blaze::DynamicMatrix<ElementType_t<MT>, SO> tmpind;
    size_t i;
    for(i = 0; i < nc;) {
        unsigned nr = std::min(size_t(8), nc - i);
        tmpind = trans(blaze::submatrix(blaze::rows(dr, rs.data(), rs.size()), 0, i * nr, rs.size(), nr));
        for(unsigned j = 0; j < nr; ++j) {
            auto r(blaze::row(tmpind, j));
            shared::sort(r.begin(), r.end());
            rr[i + j] = odd ? r[hlf]: ElementType_t<MT>(0.5) * (r[hlf] + r[hlf + 1]);
        }
        i += nr;
    }
}


template<typename MT, bool SO, typename VT2, bool TF2, typename FT=CommonType_t<ElementType_t<MT>, ElementType_t<VT2>>, typename IT=uint32_t>
static inline void weighted_median(const blz::Matrix<MT, SO> &data, blz::DenseVector<VT2, TF2> &ret, const FT *weights) {
    assert(weights);
    const size_t nc = (~data).columns();
    if((~ret).size() != nc) {
        (~ret).resize(nc);
    }
    if(unlikely((~data).columns() > ((uint64_t(1) << (sizeof(IT) * CHAR_BIT)) - 1)))
        throw std::runtime_error("Use a different index type, there are more features than fit in IT");
    const size_t nr = (~data).rows();
    auto pairs = std::make_unique<std::pair<ElementType_t<MT>, IT>[]>(nr);
    std::unique_ptr<FT[]> cw(new FT[nr]); //
    for(size_t i = 0; i < nc; ++i) {
        auto col = column(~data, i);
        for(size_t j = 0; j < nr; ++j)
            pairs[j] = {col[j], j};
        shared::sort(pairs.get(), pairs.get() + nr);
        FT wsum = 0., maxw = -std::numeric_limits<FT>::max();
        IT maxind = -0;
        for(size_t j = 0; j < nr; ++j) {
           auto neww = weights[pairs[j].second];
           wsum += neww, cw[j] = wsum;
           if(neww > maxw) maxw = neww, maxind = j;
        }
        if(maxw > wsum * .5) {
            // Return the value of the tuple with maximum weight
            (~ret)[i] = pairs[maxind].first;
            continue;
        }
        FT mid = wsum * .5;
        auto it = std::lower_bound(pairs.get(), pairs.get() + nr, mid,
             [](std::pair<ElementType_t<MT>, IT> x, FT y)
        {
            return x.first < y;
        });
        (~ret)[i] = it->first == mid ? FT(.5 * (it->first + it[1].first)): FT(it[1].first);
    }
}


template<typename MT, bool SO, typename VT, bool TF, typename VT3=blz::CommonType_t<ElementType_t<MT>, ElementType_t<VT>>>
void l1_median(const blz::Matrix<MT, SO> &data, blz::DenseVector<VT, TF> &ret, const VT3 *weights=static_cast<VT3 *>(nullptr)) {
    if(weights)
        weighted_median(data, ret, weights);
    else
        l1_unweighted_median(data, ret);
}

template<typename MT, bool SO, typename VT, bool TF, typename Rows, typename VT3=blz::CommonType_t<ElementType_t<MT>, ElementType_t<VT>>>
void l1_median(const blz::Matrix<MT, SO> &data, blz::DenseVector<VT, TF> &ret, const Rows &rows, const VT3 *weights=static_cast<VT3 *>(nullptr)) {
    if(weights) {
        auto dr(blaze::rows(data, rows.data(), rows.size()));
        const blz::CustomVector<VT3, blaze::unaligned, blaze::unpadded> cv((VT3 *)weights, (~data).rows());
        blz::DynamicVector<VT3> selected_weights(blaze::elements(cv, rows.data(), rows.size()));
        weighted_median(dr, ret, selected_weights.data());
    } else
        l1_unweighted_median(data, rows, ret);
}

} // namespace coresets
} // namespace minocore
#endif /* FGC_KMEDIAN_H__ */
