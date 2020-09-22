#pragma once
#ifndef FGC_GRAPH_DIST_H__
#define FGC_GRAPH_DIST_H__
#include "minocore/graph/graph.h"
#include "diskmat/diskmat.h"
#include <atomic>

namespace minocore {
using diskmat::DiskMat;

namespace graph {
template<typename Graph, typename MatType, typename VType=std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>>
void fill_graph_distmat(const Graph &x, MatType &mat, const VType *sources=nullptr, bool only_sources_as_dests=false, bool all_sources=false) {
    const size_t nrows = all_sources || (sources == nullptr) ? boost::num_vertices(x)
                                                             : sources->size();
    if(only_sources_as_dests && sources == nullptr) throw std::invalid_argument("only_sources_as_dests requires sources be non-null");
    const size_t ncol = only_sources_as_dests ? sources->size(): boost::num_vertices(x);
    const typename boost::graph_traits<Graph>::vertex_iterator vertices = boost::vertices(x).first;
    assert(mat.rows() == nrows);
    assert(mat.columns() == ncol);
    if(mat.rows() != nrows || mat.columns() != ncol) {
        char buf[256];
        throw std::invalid_argument(std::string(buf, std::sprintf(buf, "mat sizes (%zu rows, %zu col) don't match output requirements (%zu/%zu)\n",
                                                                  mat.rows(), mat.columns(), nrows, ncol)));
    }
    std::atomic<size_t> rows_complete;
    rows_complete.store(0);
    if(only_sources_as_dests) {
        // require that mat have nrows * ncol
        if(sources == nullptr) throw std::invalid_argument("only_sources_as_dests requires non-null sources");
        unsigned nt = 1;
#if !defined(USE_BOOST_PARALLEL) && defined(_OPENMP)
        OMP_PRAGMA("omp parallel")
        {
            OMP_PRAGMA("omp single")
            nt = omp_get_num_threads();
        }
#endif
        blaze::DynamicMatrix<float> working_space(nt, boost::num_vertices(x));
        OMP_PFOR
        for(size_t i = 0; i < nrows; ++i) {
            unsigned rowid = 0;
            OMP_ONLY(rowid = omp_get_thread_num();)
            auto vtx = all_sources ? vertices[i]: (*sources)[i];
            auto wrow(row(working_space, rowid BLAZE_CHECK_DEBUG));
            boost::dijkstra_shortest_paths(x, vtx, boost::distance_map(&wrow[0]));
            row(mat, i BLAZE_CHECK_DEBUG) = blaze::serial(blaze::elements(wrow, sources->data(), sources->size()));
            ++rows_complete;
            const auto val = rows_complete.load();
            if((val & (val - 1)) == 0)
                std::fprintf(stderr, "Completed dijkstra for row %zu/%zu\n", val, nrows);
        }
    } else {
        assert(ncol == boost::num_vertices(x));
        OMP_PFOR
        for(size_t i = 0; i < nrows; ++i) {
            auto mr = row(*mat, i BLAZE_CHECK_DEBUG);
            auto vtx = all_sources || sources == nullptr ? vertices[i]: (*sources)[i];
            assert(vtx < boost::num_vertices(x));
            boost::dijkstra_shortest_paths(x, vtx, distance_map(&mr[0]));
            ++rows_complete;
            const auto val = rows_complete.load();
            if((val & (val - 1)) == 0)
                std::fprintf(stderr, "Completed dijkstra for row %zu/%zu\n", val, nrows);
        }
    }
}

template<typename Graph, typename VType=std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>>
DiskMat<typename Graph::edge_property_type::value_type>
graph2diskmat(const Graph &x, std::string path, const VType *sources=nullptr, bool only_sources_as_dests=false, bool all_sources=false) {
    static_assert(std::is_arithmetic<typename Graph::edge_property_type::value_type>::value, "This should be floating point, or at least arithmetic");
    using FT = typename Graph::edge_property_type::value_type;
    size_t nv = sources && only_sources_as_dests ? sources->size(): boost::num_vertices(x);
    size_t nrows = all_sources || !sources ? boost::num_vertices(x): sources->size();
    std::fprintf(stderr, "all sources: %d. nrows: %zu\n", all_sources, nrows);
    DiskMat<FT> ret(nrows, nv, path);
    fill_graph_distmat(x, ret, sources, only_sources_as_dests, all_sources);
    return ret;
}


template<typename Graph, typename VType=std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>>
blaze::DynamicMatrix<typename Graph::edge_property_type::value_type>
graph2rammat(const Graph &x, std::string, const VType *sources=nullptr, bool only_sources_as_dests=false, bool all_sources=false) {
    static_assert(std::is_arithmetic<typename Graph::edge_property_type::value_type>::value, "This should be floating point, or at least arithmetic");
    using FT = typename Graph::edge_property_type::value_type;
    size_t nv = sources && only_sources_as_dests ? sources->size(): boost::num_vertices(x);
    size_t nrows = all_sources || !sources ? boost::num_vertices(x): sources->size();
    std::fprintf(stderr, "all sources: %d. nrows: %zu\n", all_sources, nrows);
    blaze::DynamicMatrix<FT>  ret(nrows, nv);
    fill_graph_distmat(x, ret, sources, only_sources_as_dests, all_sources);
    return ret;
}


} // namespace graph
using graph::fill_graph_distmat;
using graph::graph2diskmat;
using graph::graph2rammat;

} // namespace minocore

#endif
