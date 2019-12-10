#pragma once
#include <cmath>
#include <random>
#include "graph.h"
#include "aesctr/wy.h"

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
    auto &sample_from_graph(const G &x, size_t samples_per_round, size_t iterations, C &container) {
        // Algorithm D, Thorup p.415
        using Vertex = typename G::Vertex;
        //using Vertex = graph_traits<Graph>::vertex_descriptor;
        // Let R = all nodes
        auto [start, end] = x.vertices();
        flat_hash_set<Vertex> R(start, end);
        flat_hash_set<Vertex> F;
        wy::WyRand<uint64_t, 2> rng;
        std::uniform_int_distribution<Vertex> dist;
        size_t num_el = std::distance(start, end);
        // TODO: consider using hash_set distribution for provide randomness for insertion to F.
        // Maybe replace with hash set? Idk.
        for(size_t iter = 0; iter < iterations && R.size() > 0; ++iter) {
            size_t last_size = F.size();
            // Sample ``samples_per_round'' samples.
            while(F.size() < std::min(num_el, last_size + samples_per_round)) {
                Vertex ind;
                do {ind = dist(rng);} while(!F.emplace(ind).second);
            }
            // Calculate F->R distances
            // For each R, find its nearest neighbor in F
            // Pick random t in R, remove from R all points with dist(x, F) <= dist(t, F)
            // Pick random t \in R, calculate distances, remove all R with dist(x, F) leq dist(x, R)
        }
    }
};

} // graph
