#pragma once
#include <cmath>
#include <random>
#include <thread>
#include "graph.h"
#include "blaze_adaptor.h"


namespace fgc {
using namespace shared;
namespace util {
template<typename Graph>
struct ScopedSyntheticVertex {
    using Vertex = typename boost::graph_traits<Graph>::vertex_descriptor;
    Graph &ref_;
    Vertex vtx_;
    bool cleared_ = false;
    ScopedSyntheticVertex(Graph &ref): ref_(ref), vtx_(boost::add_vertex(ref_)) {
    }
    Vertex get() const {return vtx_;}
    void clear() {
        if(!cleared_) {
            boost:: clear_vertex(vtx_, ref_);
            boost::remove_vertex(vtx_, ref_);
            cleared_ = true;
        }
    }
    ~ScopedSyntheticVertex() {
        clear();
    }
};
} // namespace util
namespace thorup {
using namespace boost;

template<typename Graph, typename BBoxContainer=std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>>
std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>
&sample_from_graph(Graph &x, size_t samples_per_round, size_t iterations,
                        std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> &container, uint64_t seed,
                        const BBoxContainer *bbox_vertices_ptr=nullptr);

template<typename Graph, typename BBoxContainer=std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>>
auto
thorup_sample(Graph &x, unsigned k, uint64_t seed, size_t max_sampled=0, BBoxContainer *bbox_vertices_ptr=nullptr) {
    using Vertex = typename boost::graph_traits<Graph>::vertex_descriptor;
    if(max_sampled == 0) max_sampled = boost::num_vertices(x);
    // Algorithm E, Thorup p.418
    assert_connected(x);
    const size_t n = bbox_vertices_ptr ? bbox_vertices_ptr->size(): boost::num_vertices(x);
    //m = boost::num_edges(x);
    const double logn = std::log2(n);
    const double eps  = 1. / std::sqrt(logn);
    size_t samples_per_round = std::ceil(21. * k * logn / eps);
    size_t iterations_per_round = std::ceil(3 * logn);
    std::fprintf(stderr, "max sampled: %zu\n", max_sampled);
    std::fprintf(stderr, "samples per round: %zu\n", samples_per_round);
    std::fprintf(stderr, "iterations per round: %zu\n", iterations_per_round);
    flat_hash_set<Vertex> samples;
    std::vector<Vertex> current_buffer;
    std::mt19937_64 mt(seed);
    for(size_t i = 0, nr = std::ceil(std::pow(logn, 1.5)); i < nr; ++i) {
        sample_from_graph(x, samples_per_round, iterations_per_round, current_buffer, mt(), bbox_vertices_ptr);
        samples.insert(current_buffer.begin(), current_buffer.end());
        current_buffer.clear();
        if(samples.size() >= max_sampled) break;
        std::fprintf(stderr, "Samples size after iter %zu/%zu: %zu\n", i, nr, samples.size());
    }
    current_buffer.assign(samples.begin(), samples.end());
    if(max_sampled < samples.size())
        current_buffer.erase(current_buffer.begin() + max_sampled, current_buffer.end());
    return current_buffer;
}

template<typename Graph, typename RNG, template<typename...> class BBoxTemplate=std::vector, typename WFT=double, typename...BBoxArgs>
std::pair<std::vector<typename graph_traits<Graph>::vertex_descriptor>,
          double>
thorup_d(Graph &x, RNG &rng, size_t nperround, size_t maxnumrounds,
         const BBoxTemplate<typename boost::graph_traits<Graph>::vertex_descriptor, BBoxArgs...> *bbox_vertices_ptr=nullptr,
         const WFT *weights=nullptr)
{
    using Vertex = typename boost::graph_traits<Graph>::vertex_descriptor;
    using edge_cost = std::decay_t<decltype(get(boost::edge_weight_t(), x, std::declval<Graph>()))>;
    assert_connected(x);
    std::vector<Vertex> R;
    if(bbox_vertices_ptr) {
#ifndef NDEBUG
        for(auto vtx: *bbox_vertices_ptr) assert(vtx < boost::num_vertices(x));
#endif
        R.assign(bbox_vertices_ptr->begin(), bbox_vertices_ptr->end());
    } else {
        R.assign(boost::vertices(x).first, boost::vertices(x).second);
    }
    std::vector<Vertex> F;
    F.reserve(std::min(nperround * 5, R.size()));
    util::ScopedSyntheticVertex<Graph> vx(x);
    auto synthetic_vertex = vx.get();
    const size_t nv = boost::num_vertices(x);
    std::unique_ptr<edge_cost[]> distances(new edge_cost[nv]);
    flat_hash_set<Vertex> vertices;
    size_t i;
    if(weights) {
        throw std::runtime_error("NotImplemented: weighted sampling from R");
        //std::discrete_distribution<Vertex>(make_weighted(
        for(i = 0; R.size() && i < maxnumrounds; ++i) {
            assert(boost::num_vertices(x) == nv);
            if(R.size() > nperround) {
                do vertices.insert(R[rng() % R.size()]); while(vertices.size() < nperround);
                F.insert(F.end(), vertices.begin(), vertices.end());
                for(const auto v: vertices) boost::add_edge(v, synthetic_vertex, 0., x);
                vertices.clear();
            } else {
                for(const auto r: R) {
                    F.push_back(r);
                    boost::add_edge(F.back(), synthetic_vertex, 0., x);
                }
                R.clear();
            }
            boost::dijkstra_shortest_paths(x, synthetic_vertex,
                                           distance_map(distances.get()));
            if(R.empty()) break;
            auto randel = R[rng() % R.size()];
            auto minv = distances[randel];
            R.erase(std::remove_if(R.begin(), R.end(), [d=distances.get(),minv](auto x) {return d[x] <= minv;}), R.end());
        }
        if(i >= maxnumrounds && R.size()) {
            // This failed. Do not use this round.
          return std::make_pair(std::move(F), std::numeric_limits<double>::max());
        }
    } else {
        for(i = 0; R.size() && i < maxnumrounds; ++i) {
            if(R.size() > nperround) {
                do vertices.insert(R[rng() % R.size()]); while(vertices.size() < nperround);
                F.insert(F.end(), vertices.begin(), vertices.end());
                for(const auto v: vertices) boost::add_edge(v, synthetic_vertex, 0., x);
                vertices.clear();
            } else {
                for(const auto r: R) {
                    F.push_back(r);
                    boost::add_edge(F.back(), synthetic_vertex, 0., x);
                }
                R.clear();
            }
            boost::dijkstra_shortest_paths(x, synthetic_vertex,
                                           distance_map(distances.get()));
            if(R.empty()) break;
            auto randel = R[rng() % R.size()];
            auto minv = distances[randel];
            R.erase(std::remove_if(R.begin(), R.end(), [d=distances.get(),minv](auto x) {return d[x] <= minv;}), R.end());
        }
        if(i >= maxnumrounds && R.size()) {
            // This failed. Do not use this round.
            return std::make_pair(std::move(F), std::numeric_limits<double>::max());
        }
        assert(boost::num_vertices(x) == nv);
    }
    double cost = 0.;
    if(bbox_vertices_ptr) {
        const size_t nboxv = bbox_vertices_ptr->size();
        if(weights) {
            OMP_PRAGMA("omp parallel for reduction(+:cost)")
            for(size_t i = 0; i < nboxv; ++i) {
                cost += weights[i] * distances[bbox_vertices_ptr->operator[](i)];
            }
        } else {
            OMP_PRAGMA("omp parallel for reduction(+:cost)")
            for(size_t i = 0; i < nboxv; ++i) {
                cost += distances[bbox_vertices_ptr->operator[](i)];
            }
        }
    } else {
        OMP_PRAGMA("omp parallel for reduction(+:cost)")
        for(size_t i = 0; i < nv - 1; ++i) {
            cost += distances[i];
        }
    }
    return std::make_pair(std::move(F), cost);
}

template<typename Graph, typename BBoxContainer=std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>>
std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>
&sample_from_graph(Graph &x, size_t samples_per_round, size_t iterations,
                   std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> &container, uint64_t seed,
                   const BBoxContainer *bbox_vertices_ptr)
{
    //using edge_descriptor = typename graph_traits<Graph>::edge_descriptor;
    //typename property_map<Graph, edge_weight_t>::type weightmap = get(edge_weight, x);
    using edge_cost = std::decay_t<decltype(get(boost::edge_weight_t(), x, std::declval<Graph>()))>;
    //
    // Algorithm D, Thorup p.415
    using Vertex = typename boost::graph_traits<Graph>::vertex_descriptor;
    //using Vertex = graph_traits<Graph>::vertex_descriptor;
    // Let R = all nodes by default, or if bbox_vertices_ptr is set, all in bbox vertices
    std::vector<Vertex> R;
    if(bbox_vertices_ptr) {
        R.assign(bbox_vertices_ptr->begin(), bbox_vertices_ptr->end());
    } else {
        auto [start, end] = boost::vertices(x);
        R.assign(start, end);
    }
    auto &F = container;
    F.reserve(std::min(R.size(), iterations * samples_per_round));
    wy::WyRand<uint64_t, 2> rng(seed);
    //size_t num_el = R.size();
    util::ScopedSyntheticVertex<Graph> vx(x);
    auto synthetic_vertex = vx.get();
    // TODO: consider using hash_set distribution for provide randomness for insertion to F.
    // Maybe replace with hash set? Idk.
    auto distances = std::make_unique<edge_cost[]>(boost::num_vertices(x));
    for(size_t iter = 0; iter < iterations && R.size() > 0; ++iter) {
        //size_t last_size = F.size();
        // Sample ``samples_per_round'' samples.
        for(size_t i = 0, e = samples_per_round; i < e && R.size(); ++i) {
            auto &r = R[rng() % R.size()];
            F.emplace_back(r);
        }
        // Add connections from R to all members of F with cost 0.
        for(const auto vertex: F)
            boost::add_edge(synthetic_vertex, vertex, 0., x);
        // Calculate F->R distances
        // (one Dijkstra call with synthetic node)
        boost::dijkstra_shortest_paths(x, synthetic_vertex,
                                       distance_map(distances.get()));
        boost::clear_vertex(synthetic_vertex, x);
        // Pick random t in R, remove from R all points with dist(x, F) <= dist(t, F)
        auto el = R[rng() % R.size()];
        auto minv = distances[el];
        VERBOSE_ONLY(std::fprintf(stderr, "minv: %f\n", minv);)
        // remove all R with dist(x, F) leq dist(x, R)
        VERBOSE_ONLY(std::fprintf(stderr, "R size before: %zu\n", R.size());)
#ifdef USE_TBB
        R.erase(std::remove_if(std::execution::par_unseq, R.begin(), R.end(), [d=distances.data(),minv](auto x) {return d[x] <= minv;}), R.end());
#elif __cplusplus > 201703uL
        std::erase_if(R, [d=distances.get(),minv](auto x) {return d[x] <= minv;});
#else
        R.erase(std::remove_if(R.begin(), R.end(), [d=distances.get(),minv](auto x) {return d[x] <= minv;}), R.end());
#endif
        VERBOSE_ONLY(std::fprintf(stderr, "R size after: %zu\n", R.size());)
    }
    VERBOSE_ONLY(std::fprintf(stderr, "num vertices: %zu\n", boost::num_vertices(x));)
    boost::clear_vertex(synthetic_vertex, x);
    std::fprintf(stderr, "size: %zu\n", container.size());
    return container;
}

template<typename Graph, typename Container>
std::pair<blz::DV<std::decay_t<decltype(get(boost::edge_weight_t(), std::declval<Graph>(), std::declval<Graph>()))>>,
          std::vector<uint32_t>>
get_costs(Graph &x, const Container &container) {
    using edge_cost = std::decay_t<decltype(get(boost::edge_weight_t(), x, std::declval<Graph>()))>;
    using Vertex = typename boost::graph_traits<Graph>::vertex_descriptor;
    const size_t nv = boost::num_vertices(x);

    util::ScopedSyntheticVertex<Graph> vx(x);
    std::vector<uint32_t> assignments(boost::num_vertices(x));
    blz::DV<edge_cost> costs(boost::num_vertices(x));
    std::vector<Vertex> p(boost::num_vertices(x));

    auto synthetic_vertex = vx.get();
    for(const auto vtx: container) {
        boost::add_edge(synthetic_vertex, vtx, 0., x);
    }
    boost::dijkstra_shortest_paths(x, synthetic_vertex,
                                   distance_map(&costs[0]).predecessor_map(&p[0]));
    typename boost::property_map<Graph, boost::vertex_index_t>::type index = get(boost::vertex_index, x);
    using v_int_t = decltype(index[p[0]]);
    flat_hash_map<v_int_t, uint32_t> pid2ind;
    auto it = container.begin();
    for(size_t i = 0; i < container.size(); ++i)
        pid2ind[index[*it++]] = i;

#if 0
    for(const auto c: container)
        std::fprintf(stderr, "c: %zu\n", size_t(c));
#endif

    // This could be slow, but whatever.
    for(size_t i = 0; i < nv; ++i) {
        auto parent = p[i], newparent = p[index[parent]];
        while(newparent != synthetic_vertex) {
            parent = newparent;
            newparent = p[index[parent]];
        }
        assert(index[parent] == parent);
        //std::fprintf(stderr, "parent: %u\n", parent);
        auto it = pid2ind.find(index[parent]);
#if! NDEBUG
        if(it == pid2ind.end()) {
            //std::fprintf(stderr, "i: %zu. parent: %zu\n", i, p[i]);
            assert(pid2ind.find(i) != pid2ind.end());
        }
#endif
        assignments[i] = it == pid2ind.end() ? pid2ind.at(i): it->second;
    }
    assignments.pop_back();
    costs.resize(nv);
    assert(costs.size() == assignments.size());
    assert(nv == costs.size());
    std::fprintf(stderr, "Total cost of solution: %g\n", blaze::sum(costs));
    return std::make_pair(std::move(costs), assignments);
}
template<typename Graph, template<typename...> class BBoxTemplate=std::vector, typename...BBoxArgs>
auto 
thorup_sample_mincost(Graph &x, unsigned k, uint64_t seed, unsigned num_iter,
    const BBoxTemplate<typename boost::graph_traits<Graph>::vertex_descriptor, BBoxArgs...> *bbox_vertices_ptr=nullptr,
    double npermult=21., double nroundmult=3.)
{
    // Modification of Thorup, wherein we run Thorup Algorithm E
    // with eps = 0.5 a fixed number of times and return the best result.
    assert_connected(x);

    static constexpr double eps = 0.5;
    wy::WyRand<uint64_t, 2> rng(seed);
    const size_t n = bbox_vertices_ptr ? bbox_vertices_ptr->size(): boost::num_vertices(x);
    const double logn = std::log2(n);
    const size_t samples_per_round = std::ceil(npermult * logn * k / eps);

    std::fprintf(stderr, "nv: %zu\n", boost::num_vertices(x));
    auto func = [&](Graph &localx){return thorup_d(localx, rng, samples_per_round, nroundmult * logn, bbox_vertices_ptr);};
    std::pair<std::vector<typename graph_traits<Graph>::vertex_descriptor>,
              double> bestsol;
    bestsol.second = std::numeric_limits<double>::max();
    OMP_PFOR
    for(unsigned i = 0; i < num_iter; ++i) {
#ifdef _OPENMP
        Graph cpy = x;
        auto next = func(cpy);
#else
        auto next = func(x);
#endif
        if(next.second == std::numeric_limits<double>::max()) {
            // This round failed.
            --i;
            continue;
        }
        if(next.second < bestsol.second) {
            OMP_CRITICAL
            {
                if(next.second < bestsol.second) {
                    std::fprintf(stderr, "Replacing old cost of %g/%zu with %g/%zu\n", bestsol.second, bestsol.first.size(), next.second, next.first.size());
                    std::swap(next, bestsol);
                }
            }
        }
    }
    auto [_, assignments] = get_costs(x, bestsol.first);
    //auto assignments(get_assignments(x, bestsol.first));
    //std::fprintf(stderr, "nv: %zu\n", boost::num_vertices(x));
    return std::make_pair(std::move(bestsol.first), std::move(assignments));
}


} // thorup

#if 0
template<typename Graph, typename RNG>
void sample_cost_full(Graph &x, RNG &rng, std::ofstream &ofs, unsigned k, unsigned nsamples=1000) {
    blaze::SmallArray<unsigned, 32> indices;
    indices.reserve(k);
    std::vector<float> costs(boost::num_vertices(x) + 1, 0.);
    double maxcost = std::numeric_limits<double>::min(),
           mincost = std::numeric_limits<double>::max(),
           meancost = 0.;
    for(unsigned i = 0; i < nsamples; ++i) {
        double cost = sample_full_costs(x, rng, k, indices, costs);
        maxcost = std::max(cost, maxcost);
        mincost = std::min(cost, mincost);
        meancost += cost;
        indices.clear();
    }
    meancost /= nsamples;
    ofs << "fullproblem\t" << boost::num_vertices(x) << '\t' << mincost << '\t' << meancost << '\t' << maxcost << '\n';
}
#endif
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

#if 0
template<typename Graph>
typename boost::graph_traits<Graph>::vertex_descriptor
goldman_1median(const Graph &x,
                const std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> &p,
                typename boost::graph_traits<Graph>::vertex_descriptor source)
{
    typename boost::graph_traits<Graph>::vertex_descriptor ret;
    throw std::runtime_error("Not implemented");
    return ret;
}
#endif
template<typename PVec, typename Graph>
std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>
parallel_goldman_1median(const PVec &p, const PVec &s, const Graph &x) {
    using Vertex = typename boost::graph_traits<Graph>::vertex_descriptor;
    std::vector<Vertex> ret;
    // 1. Make trees (1 pass, serial, or each one parses the list)
    // 2. Run goldman to get 1-medians
    // 3. ??? Profit
    std::vector<std::thread> threads;
    threads.reserve(s.size());
    throw std::runtime_error("Not implemented");
#if 0
    for(const auto source: s) {
        threads.emplace_back(goldman_1median(x, p, s));
    }
#endif
    for(auto &t: threads) t.join();
    return ret;
}


template<typename ...Args>
auto idnc(boost::adjacency_list<Args...> &x, unsigned k, uint64_t seed = 0) {
    using edge_cost = std::decay_t<decltype(get(boost::edge_weight_t(), x, std::declval<boost::adjacency_list<Args...>>()))>;
    using Vertex    = typename boost::graph_traits<boost::adjacency_list<Args...>>::vertex_descriptor;
    //using Edge      = typename boost::graph_traits<boost::adjacency_list<Args...>>::edge_descriptor;
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
        std::vector<Vertex> best_vertices = parallel_goldman_1median(p, s, x);
        // Make the tree for each subset mapping best to the current solution
        // Run the acyclic algorithm from http://www.cs.kent.edu/~dragan/ST/papers/GOLDMAN-71.pdf
        // Optimal Center Location in Simple Networks
        // This can be done in linear time with the size of the tree because it's acyclic
        // and it's acyclic because it's a shortest paths tree
#if 0
        for(auto it = start; it != end; ++it) {
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
        }
#endif
        if(last_cost <= current_cost) break;
    }
}

} // namespace tnk

} // namespace fgc
