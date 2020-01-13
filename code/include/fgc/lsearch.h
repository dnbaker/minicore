#pragma once
#ifndef FGC_LOCAL_SEARCH_H__
#define FGC_LOCAL_SEARCH_H__
#include "fgc/graph.h"
#include "fgc/diskmat.h"
#include "fgc/kcenter.h"

/*
 * In this file, we use the local search heuristic for k-median.
 * Originally described in "Local Search Heuristics for k-median and Facility Location Problems",
 * Vijay Arya, Naveen Garg, Rohit Khandekar, Adam Meyerson, Kamesh Munagala, Vinayaka Pandit
 * (http://theory.stanford.edu/~kamesh/lsearch.pdf)
 */

namespace fgc {

template<typename Graph>
DiskMat<typename Graph::edge_property_type::value_type> graph2diskmat(const Graph &x, std::string path) {
    static_assert(std::is_arithmetic<typename Graph::edge_property_type::value_type>::value, "This should be floating point, or at least arithmetic");
    using FT = typename Graph::edge_property_type::value_type;
    auto nv = boost::num_vertices(x);
    DiskMat<FT> ret(nv, nv, path);
    //std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> vertices(boost::vertexs(x).first, boost::vertexs(x).second);
    typename boost::graph_traits<Graph>::vertex_iterator vertices = boost::vertices(x).first;
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0, e = boost::num_vertices(x); i < e; ++i) {
        auto mr = row(~ret, i);
        boost::dijkstra_shortest_paths(x, vertices[i], distance_map(&mr[0]));
    }
    assert((~ret).rows() == nv && (~ret).columns() == nv);
    return ret;
}

#if 0
template<typename Mat, typename Norm>
struct MatrixIndexNorm {
    const Mat mat_;
    const Norm norm_;
    template<typename Mat>
    MatrixMetric(const Mat &mat): mat_(mat) {}
    template<typename AT>
    auto operator()(size_t i, size_t j) const {
        return mat_(i, j);
    }
};
#endif

template<typename MatType, typename WFT=float, typename IType=std::uint32_t>
struct LocalKMedSearcher {
    const MatType &mat_;
    const WFT *weights_;
    blaze::SmallArray<IType, 16> sol_;
    using SolType = blaze::SmallArray<IType, 16>;
    struct MetricFromGraph {
        const MatType &ref_;
        MetricFromGraph(const MatType &ref): ref_(ref) {}
        template<typename LH, typename RH>
        WFT operator()(const LH &lh, const RH &rh) const {
            return ref_(lh, rh);
        }
    };
    LocalKMedSearcher(const MatType &mat, unsigned k, double eps=0.01, const WFT *weights=nullptr): mat_(mat), weights_(weights) {
#if 0
template<typename Iter, typename FT=ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG, typename Norm=L2Norm>
std::vector<IT>
kcenter_greedy_2approx(Iter first, Iter end, RNG &rng, size_t k, const Norm &norm=Norm())
    // Next: replace the norm() functor in kcenter_greedy_2approx,
    // with one based on indexes only
    // Use this to use MatrixMetric, to then reuse kcenter results for this.
#endif
            wy::WyRand<IType, 2> rng(k / eps * mat.rows() + mat.columns());
            auto rowits = rowiterator(mat);
            auto approx = clustering::kcenter_greedy_2approx(rowits.begin(), rowits.end(), rng, k, 
            sol_{wy::WyRand<IType, 2>(k * mat.rows() / eps + mat.columns())() % mat.rows()};
    }
    // Steps:
    // 1. Use k-center approx for seeds
    // 2. Loop over finding candidate replacements and performing swaps.
};

} // fgc

#endif /* FGC_LOCAL_SEARCH_H__ */
