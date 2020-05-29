#pragma once
#ifndef FGC_KMEDIAN_H__
#define FGC_KMEDIAN_H__
#include "minocore/optim/kmeans.h"
#include <algorithm>

namespace minocore {
namespace coresets {
using namespace blz;


namespace detail {

struct IndexCmp {
    template<typename T>
    bool operator()(const T x, const T y) const {return x->index() > y->index();}
    template<typename T, typename IT>
    bool operator()(const std::pair<T, IT> x, const std::pair<T, IT> y) const {
        return this->operator()(x.first, y.first);
    }
};

template<typename CI, typename IT=uint32_t>
struct IndexPQ: public std::priority_queue<std::pair<CI, IT>, std::vector<std::pair<CI, IT>>, IndexCmp> {
    IndexPQ(size_t nelem) {
        this->c.reserve(nelem);
    }
    auto &getc() {return this->c;}
    const auto &getc() const {return this->c;}
    auto getsorted() const {
        auto tmp = getc();
        std::sort(tmp.begin(), tmp.end(), this->comp);
        return tmp;
    }
};

} // namespace detail

template<typename MT, bool SO, typename VT, bool TF>
void sparse_l1_unweighted_median(const blz::SparseMatrix<MT, SO> &data, blz::DenseVector<VT, TF> &ret) {
    std::fprintf(stderr, "Sparse unweighted l1 median\n");
    if((~data).rows() == 1) {
        ~ret = row(~data, 0);
        return;
    }
    using FT = blaze::ElementType_t<MT>;
    auto &ctr = ~ret;
    using CI = typename MT::ConstIterator;
    const size_t nd = (~data).columns(), nr = (~data).rows(), hlf = nr / 2, odd = nr & 1;
    detail::IndexPQ<CI, uint32_t> pq(nr);
    std::unique_ptr<CI[]> ve(new CI[nr]);
    for(unsigned i = 0; i < nr; ++i) {
        auto r(row(~data, i));
        pq.push(std::pair<CI, uint32_t>(r.begin(), i));
        ve[i] = r.end();
    }
    assert(pq.size() == (~data).rows());
    uint32_t cid = 0;
    std::vector<FT> vals;
    assert(pq.empty() || pq.top().first->index() == std::min_element(pq.getc().begin(), pq.getc().end(), [](auto x, auto y) {return x.first->index() < y.first->index();})->first->index());
    // Setting all to 0 lets us simply skip elements with the wrong number of nonzeros.
    while(pq.size()) {
        //std::fprintf(stderr, "Top index: %zu\n", pq.top().first->index());
        while(cid < pq.top().first->index()) ctr[cid++] = 0;
        if(unlikely(cid > pq.top().first->index())) {
            auto pqs = pq.getsorted();
            for(const auto v: pqs) std::fprintf(stderr, "%zu:%g\n", v.first->index(), v.first->value());
            std::exit(1);
            //throw std::runtime_error("pq is incorrectly sorted.");
        }
        while(pq.top().first->index() == cid) {
            auto pair = pq.top();
            pq.pop();
            vals.push_back(pair.first->value());
            if(++pair.first != ve[pair.second]) {
                pq.push(pair);
            } else if(pq.empty()) break;
        }
        auto &cref = ctr[cid++];
        const size_t vsz = vals.size();
        if(vsz < hlf) {
            cref = 0.;
        } else {
            shared::sort(vals.data(), vals.data() + vals.size());
            size_t idx = vals.size() - nr / 2 - 1;
            if(odd) {
                cref = vals[idx];
            } else {
                cref = (vals[idx] + vals[idx + 1]) * .5;
                //cref = (vals[idx] + vals[idx - 1]) * .5;
            }
        }
        vals.clear();
    }
    while(cid < nd) ctr[cid++] = 0;
}

template<typename MT, bool SO, typename VT, bool TF>
void l1_unweighted_median(const blz::Matrix<MT, SO> &data, blz::DenseVector<VT, TF> &ret) {
#if 0
    if constexpr(blz::IsSparseMatrix_v<MT>) {
        sparse_l1_unweighted_median(~data, ret);
        return;
    }
#endif
    std::fprintf(stderr, "Dense unweighted l1 median\n");
    assert((~ret).size() == (~data).columns());
    auto &rr(~ret);
    const auto &dr(~data);
    const bool odd = dr.rows() % 2;
    const size_t hlf = dr.rows() / 2;
    for(size_t i = 0; i < dr.columns(); ++i) {
        blaze::DynamicVector<ElementType_t<MT>, blaze::columnVector> tmpind = column(data, i); // Should do fast copying.
        shared::sort(tmpind.begin(), tmpind.end());
        rr[i] = odd ? tmpind[hlf]: ElementType_t<MT>(.5) * (tmpind[hlf - 1] + tmpind[hlf]);
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
            rr[i + j] = odd ? r[hlf]: ElementType_t<MT>(0.5) * (r[hlf - 1] + r[hlf]);
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
