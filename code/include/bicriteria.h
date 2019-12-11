#pragma once
#include <cmath>
#include <random>
#include "graph.h"
#include "aesctr/wy.h"

namespace og {
using namespace boost;

template<typename ...Args>
auto &sample_from_graph(boost::adjacency_list<Args...> &x, size_t samples_per_round, size_t iterations,
                        std::vector<typename boost::graph_traits<boost::adjacency_list<Args...>>::vertex_descriptor> &container, uint64_t seed);

template<typename... Args>
std::vector<typename boost::graph_traits<boost::adjacency_list<Args...>>::vertex_descriptor>
thorup_sample(boost::adjacency_list<Args...> &x, unsigned k, uint64_t seed) {
    using Vertex = typename boost::graph_traits<boost::adjacency_list<Args...>>::vertex_descriptor;
    // Algorithm E, Thorup p.418
    const size_t n = boost::num_vertices(x),
                 m = boost::num_edges(x);
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

template<typename...Args>
auto &sample_from_graph(boost::adjacency_list<Args...> &x, size_t samples_per_round, size_t iterations,
                        std::vector<typename boost::graph_traits<boost::adjacency_list<Args...>>::vertex_descriptor> &container, uint64_t seed)
{
    using Graph = boost::adjacency_list<Args...>;
    using edge_descriptor = typename graph_traits<Graph>::edge_descriptor;
    typename property_map<Graph, edge_weight_t>::type weightmap = get(edge_weight, x);
    using edge_weight_t = std::decay_t<decltype(get(boost::edge_weight_t(), x, std::declval<boost::adjacency_list<Args...>>()))>;
    //
    // Algorithm D, Thorup p.415
    using Vertex = typename boost::graph_traits<boost::adjacency_list<Args...>>::vertex_descriptor;
    //using Vertex = graph_traits<Graph>::vertex_descriptor;
    // Let R = all nodes
    auto [start, end] = boost::vertices(x);
    std::vector<Vertex> R(start, end);
    auto &F = container;
    F.reserve(std::min(R.size(), iterations * samples_per_round));
    wy::WyRand<uint64_t, 2> rng(seed);
    size_t num_el = R.size();
    auto synthetic_vertex = boost::add_vertex(x);
    // TODO: consider using hash_set distribution for provide randomness for insertion to F.
    // Maybe replace with hash set? Idk.
    std::vector<edge_weight_t> distances(boost::num_vertices(x));
    std::vector<Vertex> p(boost::num_vertices(x));
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
        // Pick random t in R, remove from R all points with dist(x, F) <= dist(t, F)
        auto el = R[rng() % R.size()];
        auto minv = distances[el];
        // remove all R with dist(x, F) leq dist(x, R)
        for(auto it = R.begin(), e = R.end(); it != e; ++it) {
            if(distances[*it] <= minv) {
                std::swap(*it, R.back());
                R.pop_back();
                continue; // Don't increment, new value could be smaller
            }
            ++it;
        }
    }
    boost::remove_vertex(synthetic_vertex, x);
    return container;
}

} // graph
