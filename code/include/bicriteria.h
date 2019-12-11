#pragma once
#include <cmath>
#include <random>
#include "graph.h"
#include "aesctr/wy.h"

namespace og {
namespace util {
template<typename Graph>
struct ScopedSyntheticVertex {
    using Vertex = typename boost::graph_traits<Graph>::vertex_descriptor;
    Graph &ref_;
    Vertex vtx_;
    ScopedSyntheticVertex(Graph &ref): ref_(ref), vtx_(boost::add_vertex(ref_)) {}
    Vertex get() const {return vtx_;}
    ~ScopedSyntheticVertex() {
        boost::remove_vertex(vtx_, ref_);
    }
};
}
namespace thorup {
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
    std::fprintf(stderr, "samples per round: %zu\n", samples_per_round);
    std::fprintf(stderr, "iterations per round: %zu\n", iterations_per_round);
    flat_hash_set<Vertex> samples;
    std::vector<Vertex> current_buffer;
    std::mt19937_64 mt(seed);
    for(size_t i = 0, nr = std::ceil(std::pow(logn, 1.5)); i < nr; ++i) {
        sample_from_graph(x, samples_per_round, iterations_per_round, current_buffer, mt());
        samples.insert(current_buffer.begin(), current_buffer.end());
        current_buffer.clear();
        std::fprintf(stderr, "Samples size after iter %zu/%zu: %zu\n", i, nr, samples.size());
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
    using edge_cost = std::decay_t<decltype(get(boost::edge_weight_t(), x, std::declval<boost::adjacency_list<Args...>>()))>;
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
    util::ScopedSyntheticVertex<Graph> vx(x);
    auto synthetic_vertex = vx.get();
    // TODO: consider using hash_set distribution for provide randomness for insertion to F.
    // Maybe replace with hash set? Idk.
    std::vector<edge_cost> distances(boost::num_vertices(x));
    std::vector<Vertex> p(boost::num_vertices(x));
    for(size_t iter = 0; iter < iterations && R.size() > 0; ++iter) {
        size_t last_size = F.size();
        // Sample ``samples_per_round'' samples.
        for(size_t i = 0, e = samples_per_round; i < e && R.size(); ++i) {
            auto &r = R[rng() % R.size()];
            F.emplace_back(r);
            //std::swap(r, R.back()); // Move to the back
            //R.pop_back();           // Delete
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
        std::fprintf(stderr, "R size before: %zu\n", R.size());
        R.erase(std::remove_if(R.begin(), R.end(), [d=distances.data(),minv](auto x) {return d[x] <= minv;}), R.end());
        std::fprintf(stderr, "R size after: %zu\n", R.size());
    }
    std::fprintf(stderr, "size: %zu\n", container.size());
    return container;
}

} // thorup
using namespace thorup;

namespace tnk {
// Todo, Nakamura and Kudo
// MLG '19, August 05, 2019, Anchorage, AK

template<typename T, typename A>
auto random_sample(const std::vector<T, A> &v, size_t n, uint64_t seed) {
    wy::WyRand<uint64_t, 2> gen(seed);
    std::vector<T> ret;
    if(n >= v.size()) throw 1;
    ret.reserve(n);
    while(ret.size() < n) {
        auto item = v[gen() % v.size()];
        if(std::find(ret.begin(), ret.end(), item) == ret.end())
            ret.push_back(item);
    }
    return ret;
}


template<typename ...Args>
auto idnc(boost::adjacency_list<Args...> &x, unsigned k, uint64_t seed = 0) {
    using edge_cost = std::decay_t<decltype(get(boost::edge_weight_t(), x, std::declval<boost::adjacency_list<Args...>>()))>;
    using Vertex    = typename boost::graph_traits<boost::adjacency_list<Args...>>::vertex_descriptor;
    using Edge      = typename boost::graph_traits<boost::adjacency_list<Args...>>::edge_descriptor;
    using Graph     = decltype(x);
    //Iteratively Decreasing NonCentrality
    std::mt19937_64 mt(seed);
    // TODO: check to make sure it's either bidir or undir, not dir
    auto [start, end] = boost::vertices(x);

    std::vector<Vertex> R(start, end);
    std::vector<Vertex> sp = random_sample(R, k, mt()); // S' [IDNC:1]
    std::vector<Vertex> s;                              // S  [IDNC:3]
    s.reserve(k);
    util::ScopedSyntheticVertex<Graph> vx(x);
    auto synthetic_vertex = vx.get();
    edge_cost last_cost = std::numeric_limits<edge_cost>::max(), current_cost = last_cost;
    

    // For holding output of SPF / Dijkstra with multiple roots
    boost::clear_vertex(synthetic_vertex, x);
    std::vector<edge_cost> distances(boost::num_vertices(x));
    std::vector<Vertex> p(boost::num_vertices(x));

    for(const auto vertex: sp)
        boost::add_edge(synthetic_vertex, vertex, 0., x);
    boost::dijkstra_shortest_paths(x, synthetic_vertex,
                                   distance_map(&distances[0]).predecessor_map(&p[0]));
    last_cost = std::accumulate(distances.begin(), distances.end(), static_cast<edge_cost>(0));
    for(;;) {
        s = sp;
        sp.clear();
        std::vector<Vertex> best_vertices(k, static_cast<Vertex>(-1));
        for(auto it = start; it != end; ++it) {
#if 0
            Vertex cvert = *it;
            // Skip if item is a candidate center
            if(std::find(s.begin(), s.end(), cvert) != s.end()) continue;
            Vertex parent = p[cvert];
            typename std::vector<Vertex>::iterator itp;
            while((itp = std::find(s.begin(), s.end(), cvert)) == s.end());
            auto index = itp - s.begin();
            if(best_vertices[index] == static_cast<Vertex>(-1)) {
                best_vertices[index] = cvert;
            }
#endif
        }
        if(last_cost <= current_cost) break;
    }
}

} // namespace tnk

} // graph
