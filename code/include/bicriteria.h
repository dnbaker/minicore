#pragma once
#include <cmath>
#include <random>
#include "graph.h"
#include "aesctr/wy.h"

namespace og {
using namespace boost;

template<typename G>
auto &sample_from_graph(G &x, size_t samples_per_round, size_t iterations,
                        std::vector<typename G::Vertex> &container, uint64_t seed);

template<typename G>
auto thorup_sample(G &x, unsigned k, uint64_t seed) {
    using Vertex = typename G::Vertex;
    // Algorithm E, Thorup p.418
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
    current_buffer.assign(samples.begin(), samples.end());
    return current_buffer;
}

template<typename G>
auto &sample_from_graph(G &x, size_t samples_per_round, size_t iterations,
                        std::vector<typename G::Vertex> &container, uint64_t seed)
{
    using edge_weight_t = std::decay_t<decltype(x[*(boost::edges(x).first)])>;
    //
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
    auto synthetic_vertex = boost::add_vertex(x);
    // TODO: consider using hash_set distribution for provide randomness for insertion to F.
    // Maybe replace with hash set? Idk.
    std::vector<edge_weight_t> distances(x.num_vertices());
    std::vector<Vertex> p(x.num_vertices());
    for(size_t iter = 0; iter < iterations && R.size() > 0; ++iter) {
        size_t last_size = F.size();
        // Sample ``samples_per_round'' samples.
        for(size_t i = 0, e = std::min(samples_per_round, num_el - F.size()); i < e; ++i) {
            auto &r = R[rng() % R.size()];
            F.emplace_back(r);
            std::swap(r, R.back()); // Move to the back
            R.pop_back();           // Delete
        }
        // Add connections from R to all members of F with cost 0.
        boost::clear_vertex(synthetic_vertex, x);
        for(const auto vertex: F)
            boost::add_edge(synthetic_vertex, vertex, 0., x);
        // Calculate F->R distances
        // (one Dijkstra call with synthetic node)
        boost::dijkstra_shortest_paths(x, synthetic_vertex,
                                       distance_map(&distances[0]).predecessor_map(&p[0]));
#if 0
                                       predecessor_map(boost::make_iterator_property_map(p.data(), boost::get(boost::vertex_index, x))).
                                       distance_map(boost::make_iterator_property_map(&distances[0], boost::get(boost::vertex_index, x))));
#endif
        auto el = R[rng() % R.size()];
        auto minv = distances[el];
        for(auto it = R.begin(), e = R.end(); it != e; ++it) {
            if(distances[*it] <= minv) {
                std::swap(*it, R.back());
                R.pop_back();
                continue; // Don't increment, new value could be smaller
            }
            ++it;
        }
        // For each R, find its nearest neighbor in F
        // Pick random t in R, remove from R all points with dist(x, F) <= dist(t, F)
        // Pick random t \in R, calculate distances, remove all R with dist(x, F) leq dist(x, R)
    }
    boost::remove_vertex(synthetic_vertex, x);
    return container;
}

} // graph
