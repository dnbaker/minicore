#pragma once
#include <cmath>
#include <random>
#include <thread>
#include "graph.h"
#include "blaze_adaptor.h"
#include <cassert>
#include "fastiota/fastiota_ho.h"


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

template<typename Graph, typename RNG, template<typename...> class BBoxTemplate=std::vector, typename WType=uint32_t, typename...BBoxArgs>
std::pair<std::vector<typename graph_traits<Graph>::vertex_descriptor>,
          double>
thorup_d(Graph &x, RNG &rng, size_t nperround, size_t maxnumrounds,
         const BBoxTemplate<typename boost::graph_traits<Graph>::vertex_descriptor, BBoxArgs...> *bbox_vertices_ptr=nullptr,
         const WType *weights=nullptr)
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
    std::unique_ptr<Vertex[]> pmap;
    size_t i;
    if(weights) {
        if(!bbox_vertices_ptr) throw std::runtime_error("bbox_vertices_ptr must be provided to use weights");
        flat_hash_map<Vertex, Vertex> r2wi;
        for(size_t i = 0; i < bbox_vertices_ptr->size(); ++i) {
            r2wi[R[i]] = i;
        }
        auto cdf = std::make_unique<WType[]>(R.size());
        pmap.reset(new Vertex[boost::num_vertices(x)]);
        for(i = 0; R.size() && i < maxnumrounds; ++i) {
            const size_t rsz = R.size();
            std::partial_sum(R.data(), R.data() + rsz,
                           cdf.get(), [weights,&r2wi](auto csum, auto newv){return csum + weights[r2wi[newv]];});
            std::uniform_real_distribution<float> urd;
            auto weighted_select = [&]() {
                return std::lower_bound(cdf.get(), cdf.get() + rsz, cdf[rsz - 1] * urd(rng)) - cdf.get();
            };
            if(cdf[rsz - 1] > nperround) {
                WType sampled_sum = 0;
                do {
                    auto v = R[weighted_select()];
                    if(vertices.find(v) != vertices.end()) continue;
                    vertices.insert(v);
                    sampled_sum += weights[r2wi[v]];
                } while(sampled_sum < nperround);
                F.insert(F.end(), vertices.begin(), vertices.end());
                for(const auto v: vertices) boost::add_edge(v, synthetic_vertex, 0., x);
                vertices.clear();
            } else {
                F.insert(F.end(), R.begin(), R.end());
                for(const auto r: R)
                    boost::add_edge(r, synthetic_vertex, 0., x);
                R.clear();
            }
            boost::dijkstra_shortest_paths(x, synthetic_vertex,
                                           distance_map(distances.get()).predecessor_map(pmap.get()));
            if(R.empty()) break;
            auto randel = weighted_select();
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
                    boost::add_edge(r, synthetic_vertex, 0., x);
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
            flat_hash_map<Vertex, Vertex> Fmap;
            for(size_t i = 0; i < F.size(); ++i) Fmap[F[i]] = i;
            OMP_PRAGMA("omp parallel for reduction(+:cost)")
            for(size_t i = 0; i < nboxv; ++i) {
                const auto cost_inc = weights[i] * distances[bbox_vertices_ptr->operator[](i)];
                cost += cost_inc;
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
        R.erase(std::remove_if(std::execution::par_unseq, R.begin(), R.end(), [d=distances.get(),minv](auto x) {return d[x] <= minv;}), R.end());
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

    // This could be slow, but whatever.
    for(size_t i = 0; i < nv; ++i) {
        auto parent = p[i], newparent = p[index[parent]];
        while(newparent != synthetic_vertex) {
            parent = newparent;
            newparent = p[index[parent]];
        }
        assert(index[parent] == parent);
        auto it = pid2ind.find(index[parent]);
        assert(it != pid2ind.end() || pid2ind.find(i) != pid2ind.end());
        assignments[i] = it == pid2ind.end() ? pid2ind.at(i): it->second;
    }
    assignments.pop_back();
    costs.resize(nv);
    assert(costs.size() == assignments.size());
    assert(nv == costs.size());
    std::fprintf(stderr, "Total cost of solution: %g\n", blaze::sum(costs));
    return std::make_pair(std::move(costs), assignments);
}

template<typename Graph, template<typename...> class BBoxTemplate=std::vector, typename WeightType=uint32_t, typename...BBoxArgs>
auto
thorup_sample_mincost(Graph &x, unsigned k, uint64_t seed, unsigned num_iter,
    const BBoxTemplate<typename boost::graph_traits<Graph>::vertex_descriptor, BBoxArgs...> *bbox_vertices_ptr=nullptr,
    const WeightType *weights=nullptr,
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
    auto func = [&](Graph &localx) {
        //auto wworking = wcopy;
        return thorup_d(localx, rng, samples_per_round, nroundmult * logn, bbox_vertices_ptr, weights);
    };
    std::pair<std::vector<typename graph_traits<Graph>::vertex_descriptor>,
              double> bestsol;
    bestsol.second = std::numeric_limits<double>::max();
    OMP_PFOR
    for(unsigned i = 0; i < num_iter; ++i) {
        OMP_ELSE(Graph, Graph &) cpy(x);
        auto next = func(cpy);
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
    assert(assignments.size() == boost::num_vertices(x));
    //auto assignments(get_assignments(x, bestsol.first));
    //std::fprintf(stderr, "nv: %zu\n", boost::num_vertices(x));
    return std::make_pair(std::move(bestsol.first), std::move(assignments));
}

template<typename Con, typename IType=std::uint32_t, typename VertexContainer>
blz::DV<IType> histogram_assignments(const Con &c, unsigned ncenters, const VertexContainer &vtces) {
    const size_t n = std::size(vtces);
    blz::DV<IType>ret(ncenters, static_cast<IType>(0));
    OMP_PFOR
    for(size_t i = 0; i < n; ++i) {
        OMP_ATOMIC
        ++ret.at(c[vtces[i]]);
    }
#ifndef NDEBUG
    std::fprintf(stderr, "Sum of center counts: %u\n", blaze::sum(ret));
#endif
    return ret;
}

/*
 *
 *
 *
 * thorup_sample_mincost_with_weights performs sets of Thorup D ``num_trials'' in a row,
 * selecting the one with lowest cost, performing this iteration ``num_iter'' times.
 * This way, we are quite likely to have among the best solutions,
 * but with smaller and smaller set sizes.
 * It could be made faster, certainly, but it is very complicated and easy to do incorrectly.
 *
 */

template<typename Graph, template<typename...> class BBoxTemplate=std::vector, typename WeightType=uint32_t, typename...BBoxArgs>
auto
thorup_sample_mincost_with_weights(Graph &x, unsigned k, uint64_t seed,
                                   unsigned num_trials, unsigned num_iter,
    const BBoxTemplate<typename boost::graph_traits<Graph>::vertex_descriptor, BBoxArgs...> *bbox_vertices_ptr=nullptr,
    WeightType *weights=nullptr,
    double npermult=21., double nroundmult=3.)
{
    auto firstset = thorup_sample_mincost(x, k, seed, num_trials, bbox_vertices_ptr, weights, npermult, nroundmult);
    BBoxTemplate<typename boost::graph_traits<Graph>::vertex_descriptor, BBoxArgs...> bbcpy;
    if(!bbox_vertices_ptr) {
        bbcpy.assign(boost::vertices(x).first, boost::vertices(x).second);
        bbox_vertices_ptr = &bbcpy;
    }
    auto ccounts = histogram_assignments(firstset.second, firstset.first.size(), *bbox_vertices_ptr);
    assert(firstset.first.size());
#ifndef NDEBUG
    std::fprintf(stderr, "sum ccounts before anything: %u/%zu\n", blaze::sum(ccounts), ccounts.size());
    auto check_sum = [&](const auto &countcontainer) {return sum(countcontainer) == (bbox_vertices_ptr ? bbox_vertices_ptr->size(): boost::num_vertices(x));};
    assert(check_sum(ccounts));
#endif
    for(unsigned i = 1; i < num_iter; ++i) {
        assert(ccounts.size() == firstset.first.size());
        auto ccountcpy = ccounts;
        assert(check_sum(ccountcpy));
        VERBOSE_ONLY(std::fprintf(stderr, "Starting thorup sample mincost with set of elements %zu in size\n", ccountcpy.size());)
        auto nextset = thorup_sample_mincost(x, k, seed + 1, num_trials, &(firstset.first), ccountcpy.data(), npermult, nroundmult);
        ccountcpy = histogram_assignments(nextset.second, nextset.first.size(), *bbox_vertices_ptr);
        assert(ccountcpy.size() == nextset.first.size());
        ccounts = std::move(ccountcpy);
        std::swap(firstset, nextset);
    }
    return firstset;
}

/*
 * Calculates facility centers, costs, and the facility ID to which each point in the dataset is assigned.
 * This could be made iterative by:
 *  1. Performing one iteration.
 *  2. Use the selected points F as the new set of points (``npoints''), with weight = |C_f| (number of cities assigned to facility f)
 *  3. Wrap the previous oracle in another oracle that maps indices within F to the original data
 *  4. Performing the next iteration
 */
template<typename Oracle,
         typename FT=std::decay_t<decltype(std::declval<Oracle>()(0,0))>,
         typename WFT=FT,
         typename IT=uint32_t
        >
std::tuple<std::vector<IT>, blz::DV<FT>, std::vector<IT>>
oracle_thorup_d(const Oracle &oracle, size_t npoints, unsigned k, const WFT *weights=static_cast<const WFT *>(nullptr), double npermult=21, double nroundmult=3, double eps=0.5, uint64_t seed=1337)
{
    const FT total_weight = weights ? static_cast<FT>(blz::sum(blz::CustomVector<WFT, blz::unaligned, blz::unpadded>((WFT *)weights, npoints)))
                                    : static_cast<FT>(npoints);
    size_t nperround = npermult * k * std::log(total_weight) / eps;
#if VERBOSE_AF
    std::fprintf(stderr, "npoints: %zu. total weight: %g. nperround: %zu. Weights? %s\n",
                 npoints, total_weight, nperround, weights ? "true": "false");
#endif

    wy::WyRand<IT, 2> rng(seed);
    blz::DV<FT> mincosts(npoints, std::numeric_limits<FT>::max());   // minimum costs per point
    std::vector<IT> minindices(npoints, IT(-1)); // indices to which points are assigned
    size_t nr = npoints; // Manually managing count
    std::unique_ptr<IT[]> R(new IT[npoints]);
    fastiota::iota(R.get(), npoints, 0);
    std::vector<IT> F;
    shared::flat_hash_set<IT> tmp;
    std::vector<IT> current_batch;
    std::unique_ptr<FT[]> cdf(new FT[nr]);
    std::uniform_real_distribution<WFT> urd;
    auto weighted_select = [&]() {
        return std::lower_bound(cdf.get(), cdf.get() + nr, cdf[nr - 1] * urd(rng)) - cdf.get();
    };
    size_t rounds_to_do = std::ceil(nroundmult * std::log(total_weight));
    //std::fprintf(stderr, "rounds to do: %zu\n", rounds_to_do);
    while(rounds_to_do--) {
        // Sample points not yet added and calculate un-calculated distances
        if(!weights && nr <= nperround) {
            //std::fprintf(stderr, "Adding all\n");
            F.insert(F.end(), R.get(), R.get() + nr);
            for(auto it = R.get(), eit = R.get() + nr; it < eit; ++it) {
                auto v = *it;
                //std::fprintf(stderr, "Adding index %zd/value %u\n", it - R.get(), v);
                mincosts[v] = 0.;
                minindices[v] = v;
                for(size_t j = 0; j < npoints; ++j) {
                    if(j != v && mincosts[j] != 0.) {
                        if(auto score = oracle(v, j);score < mincosts[j]) {
                            mincosts[j] = score;
                            minindices[j] = v;
                        }
                    }
                }
            }
            nr = 0;
        } else {
            // Sample new points, either at random
            if(!weights) {
                //std::fprintf(stderr, "Uniformly sampling to fill tmp\n");
                while(tmp.size() < nperround) {
                    tmp.insert(rng() % nr);
                }
            // or weighted
            } else {
                std::partial_sum(R.get(), R.get() + nr, cdf.get(), [weights](auto csum, auto newv) {
                    return csum + weights[newv];
                });
#if 0
                for(size_t i = 0; i < nr; ++i) {
                    std::fprintf(stderr, "%zu|%g|%g%%\n", i, cdf[i], cdf[i] * 100. / cdf[nr - 1]);
                }
#endif
                if(cdf[nr - 1] <= nperround) {
                    //std::fprintf(stderr, "Adding the rest, nr = %zu, cdf[nr - 1] = %g\n", nr, cdf[nr - 1]);
                    for(IT i = 0; i < nr; ++i) tmp.insert(i);
                } else {
                    WFT weight_so_far = 0;
                    size_t sample_count = 0;
                    while(weight_so_far < nperround && tmp.size() < nr) {
                        ++sample_count;
                        auto ind = weighted_select();
                        if(tmp.find(ind) != tmp.end()) continue;
                        tmp.insert(ind);
                        weight_so_far += weights[R[ind]];
                        //std::fprintf(stderr, "tmp size after growing: %zu. nr: %zu. sample count: %zu. Current weight: %g. Desired weight: %zu\n", tmp.size(), nr, sample_count, weight_so_far, size_t(nperround));
                    }
#if 0
                    std::fprintf(stderr, "Took %zu samples to get %zu items of total weight %g\n", sample_count, tmp.size(), weight_so_far);
#endif
                }
#if 0
                std::fprintf(stderr, "Sampled %zu items of total weight %0.12g\n", tmp.size(),
                             std::accumulate(tmp.begin(), tmp.end(), 0., [&](auto y, auto x) {return y + weights[R[x]];}));
#endif
            }
            // Update F, R, and mincosts/minindices
            current_batch.assign(tmp.begin(), tmp.end());
            tmp.clear();
            for(const auto item: current_batch)
                F.push_back(R[item]);
            shared::sort(current_batch.begin(), current_batch.end(), std::greater<>());
            for(const auto v: current_batch) {
                auto actual_index = R[v];
                minindices[actual_index] = actual_index;
                mincosts[actual_index] = 0.;
                std::swap(R[v], R[--nr]);
                for(size_t j = 0; j < npoints; ++j) {
                    if(j != actual_index) {
                        if(auto oldcost = mincosts[j]; oldcost != 0.) {
                            auto newcost = oracle(actual_index, j);
                            if(newcost < oldcost) {
                                mincosts[j] = newcost;
                                minindices[j] = actual_index;
                            }
                        }
                    }
                }
            }
        }
        // Select pivot and remove others.
        if(nr == 0) break;
        unsigned pivot_index;
        if(weights) {
            std::partial_sum(R.get(), R.get() + nr, cdf.get(), [weights](auto csum, auto newv) {
                return csum + weights[newv];
            });
            pivot_index = weighted_select();
        } else {
            pivot_index = rng() % nr;
        }
        //auto &pivot = R[pivot_index];
        const FT pivot_mincost = mincosts[R[pivot_index]];
        for(auto it = R.get() + nr, e = R.get(); --it >= e;)
            if(auto &v = *it; mincosts[v] <= pivot_mincost)
                std::swap(v, R[--nr]);
    }
#if VERBOSE_AF
    FT final_total_cost = 0.;
    for(size_t i = 0; i < mincosts.size(); ++i) {
        final_total_cost += weights ? mincosts[i] * weights[i]: mincosts[i];
    }
    std::fprintf(stderr, "[LINE %d] Returning solution with mincosts [%zu] and minindices [%zu] with F of size %zu/%zu and final total cost %g for weight %g.\n", __LINE__, mincosts.size(), minindices.size(), F.size(), npoints, final_total_cost, total_weight);
#endif
#if 0
    for(size_t i = 0; i < mincosts.size(); ++i) {
        std::fprintf(stderr, "ID %zu has %g as mincost and %u as minind\n", i, mincosts[i], minindices[i]);
    }
#endif
    return {F, mincosts, minindices};
}

template<typename Oracle, typename IT=uint32_t>
struct OracleWrapper {
    const Oracle &oracle_;
    std::vector<IT> lut_;
public:
    template<typename Container>
    OracleWrapper(const Oracle &oracle, const Container &indices): OracleWrapper(oracle, indices.begin(), indices.end()) {
    }
    template<typename It>
    OracleWrapper(const Oracle &oracle, It start, It end):
        oracle_(oracle), lut_(start, end) {}

    INLINE decltype(auto) operator()(size_t i, size_t j) const {
       return oracle_(lookup(i), lookup(j));
    }

    IT lookup(size_t idx) const {
#ifndef NDEBUG
        return lut_.at(idx);
#else
        return lut_[idx];
#endif
    }
};

template<typename Oracle, typename Container>
auto
make_oracle_wrapper(const Oracle &o, const Container &indices) {
    using IT = std::decay_t<decltype(*indices.begin())>;
    return OracleWrapper<Oracle, IT>(o, indices);
}
template<typename Oracle, typename It>
auto
make_oracle_wrapper(const Oracle &o, It start, It end) {
    using IT = std::decay_t<decltype(*start)>;
    return OracleWrapper<Oracle, IT>(o, start, end);
}

/*
 * Note: iterated_oracle_thorup_d uses the cost *according to the weighted data* from previous iterations,
 * not the cost of the current solution against the original data when selecting which
 * sub-iteration to pursue. This might be change in future iterations.
 */

template<typename Oracle,
         typename FT=std::decay_t<decltype(std::declval<Oracle>()(0,0))>,
         typename WFT=FT,
         typename IT=uint32_t
        >
std::tuple<std::vector<IT>, blz::DV<FT>, std::vector<IT>>
iterated_oracle_thorup_d(const Oracle &oracle, size_t npoints, unsigned k, unsigned num_iter=3, unsigned num_sub_iter=8,
                         const WFT *weights=static_cast<const WFT *>(nullptr), double npermult=21, double nroundmult=3, double eps=0.5, uint64_t seed=1337)
{
    auto getw = [weights](size_t index) {
        return weights ? weights[index]: static_cast<WFT>(1.);
    };
#if !NDEBUG
    const FT total_weight = weights ? blz::sum(blz::CustomVector<WFT, blz::unaligned, blz::unpadded>((WFT *)weights, npoints))
                                    : WFT(npoints);
#endif
    wy::WyHash<uint64_t, 2> rng(seed);
    std::tuple<std::vector<IT>, blz::DV<FT>, std::vector<IT>> ret;
    auto &[centers, costs, bestindices] = ret; // Unpack for named access
    FT best_cost;
    // For convenience: a custom vector
    //                  which is empty if weights is null and full otherwise.
    {
        std::unique_ptr<blz::CustomVector<const WFT, blz::unaligned, blz::unpadded>> wview;
        if(weights) wview.reset(new blz::CustomVector<const WFT, blz::unaligned, blz::unpadded>(weights, npoints));
        auto do_thorup_sample = [&]() {
            return oracle_thorup_d(oracle, npoints, k, weights, npermult, nroundmult, eps, rng());
        };
        auto get_cost = [&](const auto &x) {
            return wview ? blz::dot(x, *wview): blz::sum(x);
        };

        // gather first set of sampled points
        ret = do_thorup_sample();
        best_cost = get_cost(costs);

        // Repeat this process a number of times and select the best-scoring set of points.
        OMP_PFOR
        for(unsigned sub_iter = 0; sub_iter < num_sub_iter; ++sub_iter) {
            auto next_sol = do_thorup_sample();
            auto &[centers2, costs2, bestindices2] = next_sol;
            auto next_cost = get_cost(costs2);
            if(next_cost < best_cost) {
                OMP_CRITICAL
                {
#ifdef _OPENMP
                    if(next_cost < best_cost)
                    // Check again after acquiring the lock in case the value has changed, but only
                    // if parallelized
#endif
                    {
                        ret = std::move(next_sol);
                        best_cost = next_cost;
                    }
                }
            }
        }
    }

    // Calculate weights for center points
    blz::DV<FT> center_weights(centers.size(), FT(0));
    shared::flat_hash_map<IT, IT> asn2id; asn2id.reserve(centers.size());
    for(size_t i = 0; i < centers.size(); asn2id[centers[i]] = i, ++i);
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0; i < npoints; ++i) {
        const auto weight = getw(i);
        auto it = asn2id.find(bestindices[i]);
        assert(it != asn2id.end());
        OMP_ATOMIC
        center_weights[it->second] += weight;
    }
#ifndef NDEBUG
    bool nofails = true;
    for(size_t i = 0; i < center_weights.size(); ++i) {
        if(center_weights[i] <= 0.) {
            std::fprintf(stderr, "weight %zu for center %u is nonpositive: %g and is a center\n", i, centers[i], center_weights[i]);
            nofails = false;
        }
    }
    assert(std::abs(blz::sum(center_weights) - total_weight) < 1e-4 ||
           !std::fprintf(stderr, "Expected sum %g, found %g\n", total_weight, blz::sum(center_weights)));
    assert(nofails);
#endif
    for(size_t iter = 0; iter < num_iter; ++iter) {
        // Setup helpers:
        auto wrapped_oracle = make_oracle_wrapper(oracle, centers); // Remapping old oracle to new points.
        auto do_iter_thorup_sample = [&]() { // Performs wrapped oracle Thorup D
            return oracle_thorup_d(wrapped_oracle, centers.size(), k, center_weights.data(), npermult, nroundmult, eps, rng());
        };
        auto get_cost = [&](const auto &x) { // Calculates the cost of a set of centers.
            return blz::dot(x, center_weights);
        };

        // Get first solution
        auto [sub_centers, sub_costs, sub_bestindices] = do_iter_thorup_sample();
        best_cost = get_cost(sub_costs);
        OMP_PFOR
        for(unsigned sub_iter = 0; sub_iter < num_sub_iter; ++sub_iter) {
            auto new_ret = do_iter_thorup_sample();
            if(auto next_cost = get_cost(std::get<1>(new_ret)); next_cost < best_cost) {
                OMP_CRITICAL
                {
#ifdef _OPENMP
                    if(next_cost < best_cost)
#endif
                    {
                        std::tie(sub_centers, sub_costs, sub_bestindices) = std::move(new_ret);
#if VERBOSE_AF
                        std::fprintf(stderr, "[subiter %zu|mainiter %u] iter ret sizes after replacing old cost %g with %g: %zu/%zu/%zu\n",
                                     iter, sub_iter, best_cost, next_cost, sub_centers.size(), sub_costs.size(), sub_bestindices.size());
#endif
                        best_cost = next_cost;
                    }
                }
            }
        }
        // reassign centers and weights
        shared::flat_hash_map<IT, IT> sub_asn2id; sub_asn2id.reserve(sub_centers.size());
        for(size_t i = 0; i < sub_centers.size(); sub_asn2id[sub_centers[i]] = i, ++i);
        assert(sub_bestindices.size() == center_weights.size() ||
                !std::fprintf(stderr, "sub_bestindices size %zu, vs expected %zu\n", sub_bestindices.size(), center_weights.size()));

        blz::DV<FT> sub_center_weights(sub_centers.size(), FT(0));
        OMP_PFOR
        for(size_t i = 0; i < sub_bestindices.size(); ++i) {
            auto weight_idx = sub_asn2id.operator[](sub_bestindices[i]); // at to bounds-check
            auto item_weight = center_weights[i];
            OMP_ATOMIC
            sub_center_weights[weight_idx] += item_weight;
        }

        DBG_ONLY(for(const auto w: sub_center_weights) assert(w > 0.);)
        assert(std::abs(blz::sum(sub_center_weights) - total_weight) <= 1.e-4);

        std::transform(sub_centers.begin(), sub_centers.end(), sub_centers.begin(),
                       [&wrapped_oracle](auto x) {return wrapped_oracle.lookup(x);});
        std::transform(sub_bestindices.begin(), sub_bestindices.end(), sub_bestindices.begin(),
                       [&wrapped_oracle](auto x) {return wrapped_oracle.lookup(x);});
        centers = std::move(sub_centers);
        center_weights = std::move(sub_center_weights);
        bestindices = std::move(sub_bestindices);
#if VERBOSE_AF
        std::fprintf(stderr, "after resizing, centers size: %zu\n", centers.size());
        std::fprintf(stderr, "after resizing, center_weights size: %zu\n", center_weights.size());
#endif
    }
    return {std::move(centers), std::move(costs), std::move(bestindices)};
}


} // thorup

using thorup::thorup_sample_mincost_with_weights;
using thorup::thorup_sample_mincost;
using thorup::sample_from_graph;
using thorup::thorup_sample;
using thorup::histogram_assignments;
using thorup::thorup_d;
using thorup::oracle_thorup_d;
using thorup::get_costs;
using thorup::iterated_oracle_thorup_d;
using thorup::oracle_thorup_d;


} // namespace fgc
