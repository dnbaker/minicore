#pragma once
#include <cmath>
#include "graph.h"

namespace graph {


template<typename S>
struct ThorupApprox {
    S container_;
    template<typename G>
    auto thorup_sample(const G &x, unsigned k) {
        using Vertex = typename G::Vertex;
        // Algorithm E, Thorup p.418
        container_.clear();
        const size_t n = x.num_vertices(),
                     m = x.num_edges();
        const double logn = std::log2(n);
        const double eps  = std::sqrt(logn);
        size_t samples_per_round = std::ceil(21. * k * logn / eps);
        size_t iterations_per_round = std::ceil(3 * logn);
        flat_hash_set<Vertex> samples;
        flat_hash_set<Vertex> current_buffer;
        for(size_t i = 0, nr = std::ceil(std::pow(logn, 1.5)); i < nr; ++i) {
            sample_from_graph(x, samples_per_round, iterations_per_round, current_buffer);
            samples.insert(current_buffer.begin(), current_buffer.end());
            current_buffer.clear();
        }
        return samples;
    }
    template<typename G, typename C>
    auto &sample_from_graph(const G &x, size_t samples, size_t iterations, C &container) {
        // Algorithm D, Thorup p.415
        using Vertex = typename G::Vertex;
        //using Vertex = graph_traits<Graph>::vertex_descriptor;
        // Let R = all nodes
        auto [start, end] = x.vertices();
        flat_hash_set<Vertex> R(start, end);
        // Maybe replace with hash set? Idk.
        for(size_t iter = 0; iter < iterations && R.size() > 0; ++iter) {
            // Sample ``samples'' samples.
            // Compute distances
            // Pick random t \in R, calculate distances, remove all R with dist(x, F) leq dist(x, R)
        }
    }
};

} // graph
