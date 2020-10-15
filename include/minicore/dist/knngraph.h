#include "minicore/graph.h"
#include "minicore/util/packed.h"
#include "minicore/dist/applicator.h"
#include "minicore/hash/hash.h"
#include <boost/graph/kruskal_min_spanning_tree.hpp>

namespace minicore {

template<typename IT=uint32_t, typename MatrixType>
std::vector<packed::pair<blaze::ElementType_t<MatrixType>, IT>> make_knns(const jsd::DissimilarityApplicator<MatrixType> &app, unsigned k) {
    using FT = blaze::ElementType_t<MatrixType>;
    static_assert(std::is_integral_v<IT>, "Sanity");
    static_assert(std::is_floating_point_v<FT>, "Sanity");

    MINOCORE_REQUIRE(std::numeric_limits<IT>::max() > app.size(), "sanity check");
    if(k > app.size()) {
        std::fprintf(stderr, "Note: make_knn_graph was provided k (%u) > # points (%zu).\n", k, app.size());
        k = app.size();
    }
    const size_t np = app.size();
    const jsd::DissimilarityMeasure measure = app.get_measure();
    std::vector<packed::pair<FT, IT>> ret(k * np);
    std::vector<unsigned> in_set(np);
    const bool measure_is_sym = distance::is_symmetric(measure);
    const bool measure_is_dist = distance::is_dissimilarity(measure);
    std::unique_ptr<std::mutex[]> locks;
    OMP_ONLY(locks.reset(new std::mutex[np]);)

    // Helper functions
    // Update
    auto update_fwd = [&](FT d, size_t i, size_t j) {
        if(in_set[i] < k) {
            OMP_ONLY(std::lock_guard<std::mutex> lock(locks[i]);)
            ret[i * k + in_set[i]] = packed::pair<FT, IT>{d, j};
            if(++in_set[i] == k) {
                if(measure_is_dist)
                    std::make_heap(ret.data() + i * k, ret.data() + (i + 1) * k, std::less<void>());
                else
                    std::make_heap(ret.data() + i * k, ret.data() + (i + 1) * k, std::greater<void>());
            }
        } else {
            auto cmp = [&](auto d) {return (ret[i * k].first < d) ^ measure_is_dist;};
            auto pushpop = [&](auto d) {
                auto startp = &ret[i * k];
                auto stopp = startp + k;
                if(measure_is_dist) std::pop_heap(startp, stopp, std::less<void>());
                else                std::pop_heap(startp, stopp, std::greater<void>());
                ret[(i + 1) * k - 1] = packed::pair<FT, IT>{d, j};
                if(measure_is_dist) std::push_heap(startp, stopp, std::less<void>());
                else                std::push_heap(startp, stopp, std::greater<void>());
            };
            if(cmp(d)) {
                OMP_ONLY(std::lock_guard<std::mutex> lock(locks[i]);)
                {
                    OMP_ONLY(if(cmp(d)))
                        pushpop(d);
                }
            }
        }
    };
    auto update_both = [&](auto d, auto i, auto j) ALWAYS_INLINE {
        update_fwd(d, i, j); update_fwd(d, j, i);
    };

    // Sort
    auto perform_sort = [&](auto ptr) {
        auto end = ptr + k;
        if(measure_is_dist)
            shared::sort(ptr, end, std::less<>());
        else
            shared::sort(ptr, end, std::greater<>());
        ptr = end;
    };
    auto ptr = ret.data();
    if(measure_is_sym) {
        OMP_PFOR
        for(size_t i = 0; i < np; ++i) {
            for(size_t j = i + 1; j < np; ++j) {
                update_both(app(i, j), i, j);
            }
            perform_sort(ptr);
        }
    } else {
        OMP_PFOR
        for(size_t i = 0; i < np; ++i) {
            for(size_t j = 0; j < np; ++j) {
                update_fwd(app(i, j), i, j);
            }
            perform_sort(ptr);
        }
    }
    std::fprintf(stderr, "Created knn graph for k = %u and %zu points\n", k, np);
    return ret;
}

template<typename IT=uint32_t, typename MatrixType, typename Hasher, typename IT2=IT, typename KT>
std::vector<packed::pair<blaze::ElementType_t<MatrixType>, IT>>
make_knns_by_lsh(const jsd::DissimilarityApplicator<MatrixType> &app, hash::LSHTable<Hasher, IT2, KT> &table, unsigned k, unsigned maxlshcmp=0)
{
    if(!maxlshcmp) maxlshcmp = 10 * k;
    using FT = blaze::ElementType_t<MatrixType>;
    static_assert(std::is_integral_v<IT>, "Sanity");
    static_assert(std::is_floating_point_v<FT>, "Sanity");

    MINOCORE_REQUIRE(std::numeric_limits<IT>::max() > app.size(), "sanity check");
    if(k > app.size()) {
        std::fprintf(stderr, "Note: make_knn_graph was provided k (%u) > # points (%zu).\n", k, app.size());
        k = app.size();
    }
    const size_t np = app.size();
    const jsd::DissimilarityMeasure measure = app.get_measure();
    std::vector<packed::pair<FT, IT>> ret(k * np);
    std::vector<unsigned> in_set(np);
    const bool measure_is_sym = dist::is_symmetric(measure);
    const bool measure_is_dist = dist::is_dissimilarity(measure);
    OMP_ONLY(std::unique_ptr<std::mutex[]> locks(new std::mutex[np]);)
    table.add(app.data());
    table.sort();


    auto update_fwd = [&](FT d, size_t i, size_t j) {
        if(in_set[i] < k) {
            OMP_ONLY(std::lock_guard<std::mutex> lock(locks[i]);)
            ret[i * k + in_set[i]] = packed::pair<FT, IT>{d, j};
            if(++in_set[i] == k) {
                if(measure_is_dist)
                    std::make_heap(ret.data() + i * k, ret.data() + (i + 1) * k, std::less<void>());
                else
                    std::make_heap(ret.data() + i * k, ret.data() + (i + 1) * k, std::greater<void>());
            }
        } else {
            auto cmp = [&](auto d) {return measure_is_dist ? (ret[i * k].first > d) : (ret[i * k].first < d);};
            auto pushpop = [&](auto d) {
                auto startp = &ret[i * k];
                auto stopp = startp + k;
                std::pop_heap(startp, stopp, std::less<void>());
                ret[(i + 1) * k - 1] = packed::pair<FT, IT>{d, j};
                std::push_heap(startp, stopp, std::less<void>());
            };
            if(cmp(d)) {
                OMP_ONLY(std::lock_guard<std::mutex> lock(locks[i]);)
                {
                    OMP_ONLY(if(cmp(d)))
                        pushpop(d);
                }
            }
        }
    };

    // Sort
    auto ptr = ret.data();
    MINOCORE_VALIDATE(maxlshcmp <= k);
    OMP_PFOR
    for(size_t i = 0; i < np; ++i) {
        auto tk = table.topk(row(app.data(), i, blaze::unchecked), maxlshcmp);
        for(const auto &pair: tk) {
            if(pair.first != i) {
                auto d = app(i, pair.first);
                update_fwd(d, i, pair.first);
                update_fwd(d, pair.first, i);
            }
        }
    }
    size_t number_exhaustive = 0;
    for(size_t i = 0; i < np; ++i) {
        if(in_set[i] >= k) continue;
        ++number_exhaustive;
        std::fprintf(stderr, "Warning: LSH table returned < k (%d) neighbors (only %d compared). Performing exhaustive comparisons for item %zu\n",
                     k, in_set[i], i);
        OMP_PFOR
        for(size_t j = (measure_is_sym ? i + 1: size_t(0)); j < np; ++j) {
            if(unlikely(j == i)) continue;
            auto d = app(i, j);
            update_fwd(d, i, j);
            update_fwd(d, j, i);
        }
    }
    if(number_exhaustive)
        std::fprintf(stderr, "Performed quadratic distance comparisons with %zu/%zu items\n",
                     number_exhaustive, np);
    std::fprintf(stderr, "Created knn graph for k = %u and %zu points\n", k, np);
    return ret;
}

template<typename IT=uint32_t, typename FT=float>
auto knns2graph(const std::vector<packed::pair<FT, IT>> &knns, size_t np, bool mutual=true, bool symmetric=true) {
    MINOCORE_REQUIRE(knns.size() % np == 0, "sanity");
    MINOCORE_REQUIRE(knns.size(), "nonempty");
    unsigned k = knns.size() / np;
    MINOCORE_REQUIRE(knns.size() == np * k, "sanity");
    std::cerr << "k: " << k << "knns: " << knns.size() << (mutual ? "mutual" : "absolute") << '\n';
    graph::Graph<boost::undirectedS, FT> ret(np);
    std::fprintf(stderr, "produced graph\n");
    for(size_t i = 0; i < np; ++i) {
        auto p = &knns[i * k];
        SK_UNROLL_8
        for(unsigned j = 0; j < k; ++j) {
            if(mutual) {
                if(symmetric) {
                    if(p[j].first > knns[k * (p[j].second + 1) - 1].first)
                        continue;
                } else {
                    // More expensive (O(k) vs O(1)), but does not require the assumption of symmetry.
                    if(auto start = knns.data() + p[j].second * k, stop = start + k;
                       std::find_if(start, stop, [i](auto x) {return x.second == i;}) == stop)
                        continue;
                }
            }
            boost::add_edge(i, static_cast<size_t>(p[j].second), p[j].first, ret);
        }
        std::fprintf(stderr, "%zu/%zu\n", i + 1, np);
    }
    return ret;
}

template<typename IT=uint32_t, typename MatrixType>
auto make_knn_graph(const jsd::DissimilarityApplicator<MatrixType> &app, unsigned k, bool mutual=true) {
    return knns2graph(make_knns(app, k), app.size(), mutual, dist::is_symmetric(app.get_measure()));
}

template<typename IT=uint32_t, typename Graph>
auto knng2mst(const Graph &gr) {
    std::vector<typename boost::graph_traits<Graph>::edge_descriptor> ret;
    ret.reserve(boost::num_vertices(gr) * 1.5);
    boost::kruskal_minimum_spanning_tree(gr, std::back_inserter(ret));
    return ret;
}

#if 0
template<typename IT=uint32_t, typename MatrixType>
auto perform_rcc(const jsd::DissimilarityApplicator<MatrixType> &app, unsigned k, bool mutual=true, size_t niter=100) {
    using FT = blaze::ElementType_t<MatrixType>;
    auto graph = make_knn_graph(app, k);
    auto mst = knng2mst(graph);
    using eh_t = std::conditional_t<sizeof(IT) <= 4, uint64_t, std::pair<IT, IT> >;
    shared::flat_hash_set<eh_t> edges;
    auto add_edge = [&](auto edge) {
        if constexpr(sizeof(IT) <= 4) {
            uint64_t encoded = (uint64_t(boost::source(edge, graph)) << 32) | boost::target(edge, graph);
            edges.insert(encoded);
        } else {
            edges.insert(eh_t(boost::source(edge, graph), boost::target(edge, graph)));
        }
    };
    for(const auto edge: mst) {
        add_edge(edge);
    }
    for(const auto edge: graph.edges()) {
        add_edge(edge);
    }
    size_t nedges = edges.size();
    std::unique_ptr<IT[]> lhp(new IT[nedges]), rhp(new IT[nedges]);
    size_t i = 0;
    for(const auto &e: edges) {
        lhp[i] = e.first;
        rhp[i] = e.second;
        ++i;
    }
    // Free unneeded memory
    { shared::flat_hash_set<eh_t> tmp(std::move(edges)); }
    const double xi = blaze::norm(app.data());
    blaze::DynamicMatrix<FT> U = app.data();
    blaze::DynamicVector<FT> lpq(nedges, 1.);
    blaze::DynamicVector<FT> epsilons = blaze::generate(nedges, [&](auto x) {
        return app(lhp[x], rhp[x]);
    });
    shared::sort(epsilons.data(), epsilons.data() + nedges);
    const int top_samples = std::minimum(250, int(std::ceil(nedges*0.01)));
    double delta = blaze::mean(blaze::subvector(epsilons, 0, top_samples));
    double eps   = blaze::mean(blaze::subvector(epsilons, 0, int(std::ceil(nedge * 0.01)));
    const double mu = 3.0 * std::pow(epsilons[nedges - 1], 2.);
    auto calculate_objective = [&]() {
        auto dat = .5 * blaze::sum(blaze::pow(app.data() - U), 2.);
        return dat;
    };
    std::vector<FT> obj;
    for(size_t iternum = 0; iternum < niter; ++iternum) {
        OMP_PFOR
        for(size_t i = 0; i < app.data().columns(); ++i) {
            lpq[i] = mu / (mu + app(lhp[i], rhp[i]));
        }
        obj.push_back(calculate_objective());
    }
}
#endif


} // minicore
