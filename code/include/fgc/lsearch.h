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

/*
 * 
 *
 *
 */

template<typename MatType, typename WFT=float, typename IType=std::uint32_t>
struct LocalKMedSearcher {
    const MatType &mat_;
    const WFT *weights_;
    blaze::SmallArray<IType, 16> sol_;
    using SolType = blaze::SmallArray<IType, 16>;
    LocalKMedSearcher(const MatType &mat, unsigned k, double eps=0.01, const WFT *weights=nullptr): mat_(mat), weights_(weights) {
    }
    // Steps:
    // 1. Use k-center approx for seeds
    // 2. Loop over finding candidate replacements and performing swaps.
};

} // fgc

#endif /* FGC_LOCAL_SEARCH_H__ */
