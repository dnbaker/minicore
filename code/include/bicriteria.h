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
    auto thorup_sample(const G &x, unsigned k, uint64_t seed=0) {
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
        std::vector<Vertex> current_buffer;
        std::mt19937_64 mt(seed);
        for(size_t i = 0, nr = std::ceil(std::pow(logn, 1.5)); i < nr; ++i) {
            sample_from_graph(x, samples_per_round, iterations_per_round, current_buffer, mt());
            samples.insert(current_buffer.begin(), current_buffer.end());
            current_buffer.clear();
        }
        current_buffer.insert(samples.begin(), samples.end());
        return current_buffer;
    }
    template<typename G>
    auto &sample_from_graph(const G &x, size_t samples_per_round, size_t iterations,
                            std::vector<typename G::Vertex> &container, uint64_t seed) {
        // Algorithm D, Thorup p.415
        using Vertex = typename G::Vertex;
        //using Vertex = graph_traits<Graph>::vertex_descriptor;
        // Let R = all nodes
        auto [start, end] = x.vertices();
        std::vector<Vertex> R(start, end);
        auto &F = container;
        F.reserve(std::min(R.size(), iterations * samples_per_round));
        wy::WyRand<uint64_t, 2> rng(seed);
        size_t num_el = R.size();
        // TODO: consider using hash_set distribution for provide randomness for insertion to F.
        // Maybe replace with hash set? Idk.
        for(size_t iter = 0; iter < iterations && R.size() > 0; ++iter) {
            size_t last_size = F.size();
            // Sample ``samples_per_round'' samples.
            for(size_t i = 0, e = std::min(samples_per_round, num_el - F.size()); i < e; ++i) {
                auto &r = R[rng() % R.size()];
                F.emplace_back(r);
                std::swap(r, R.back()); // Move to the back
                R.pop_back();           // Delete
            }
            // Calculate F->R distances
            // For each R, find its nearest neighbor in F
            // Pick random t in R, remove from R all points with dist(x, F) <= dist(t, F)
            // Pick random t \in R, calculate distances, remove all R with dist(x, F) leq dist(x, R)
        }
    }
};

} // graph
