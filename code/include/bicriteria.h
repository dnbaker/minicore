#pragma once
#include <cmath>
#include "graph.h"

namespace graph {

template<typename S>
struct ThorupApprox {
    S container_;
    template<typename G>
    void thorup_sample(const G &x, unsigned k) {
        // Algorithm E, Thorup p.418
        container_.clear();
        const size_t n = x.num_vertices(),
                     m = x.num_edges();
        const double logn = std::log2(n);
        const double eps  = std::sqrt(logn);
        size_t samples_per_round =  21. * k * logn / eps;
        size_t iterations_per_round = 
        for(size_t i = 0, nr = std::ceil(std::pow(logn, 1.5)); i < nr; ++i) {
            sample_from_graph(x, samples_per_round, iterations_per_round);
        }
    }
    template<typename G>
    void sample_from_graph(const G &x, size_t samples, size_t iterations) {
        // Algorithm D, Thorup p.415
        using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
        // Let R = all nodes
        auto [start, end] = boost::nodes(x);
        std::vector<Vertex> R(start, end);
        // Maybe replace with hash set? Idk.
        for(size_t iter = 0; iter < iterations && R.size() > 0; ++iter) {
            // Sample ``samples'' samples.
            // Computer distances
            // Pick random t \in R, remove all R with dist(x, F) leq dist(x, R)
        }
    }
};

} // graph
