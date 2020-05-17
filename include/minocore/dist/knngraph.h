#include "minocore/graph.h"
#include "minocore/util/packed.h"
#include "minocore/dist/applicator.h"
#include <boost/graph/kruskal_min_spanning_tree.hpp>

namespace minocore {

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
    const bool measure_is_sym = blz::detail::is_symmetric(measure);
    const bool measure_is_dist = measure_is_dist;
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
            auto cmp = [&](auto d) {return measure_is_dist ? (ret[i * k].first > d) : (ret[i * k].first < d);};
            auto pushpop = [&](auto d) {
                auto startp = &ret[i * k];
                auto stopp = startp + k;
                if(measure_is_dist) std::pop_heap(startp, stopp, std::less<void>());
                                    std::pop_heap(startp, stopp, std::greater<void>());
                ret[(i + 1) * k - 1] = packed::pair<FT, IT>{d, j};
                if(measure_is_dist) std::push_heap(startp, stopp, std::less<void>());
                                    std::push_heap(startp, stopp, std::greater<void>());
            };
            if(cmp(d)) {
#ifdef _OPENMP
                std::lock_guard<std::mutex> lock(locks[i]);
#endif
                {
                    OMP_ONLY(if(cmp(d)))
                        pushpop(d);
                }
            
            }
        }
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
                auto d = app(i, j);
                update_fwd(d, i, j);
                update_fwd(d, j, i);
            }
            perform_sort(ptr);
            std::fprintf(stderr, "[Symmetric:%s] Completed %zu/%zu\n", blz::detail::prob2str(measure), i + 1, np);
        }
    } else {
        OMP_PFOR
        for(size_t i = 0; i < np; ++i) {
            for(size_t j = 0; j < np; ++j) {
                update_fwd(app(i, j), i, j);
            }
            perform_sort(ptr);
            std::fprintf(stderr, "[Asymmetric:%s] Completed %zu/%zu\n", blz::detail::prob2str(measure), i + 1, np);
        }
    }
    std::fprintf(stderr, "Created knn graph for k = %u and %zu points\n", k, np);
    return ret;
}

template<typename IT=uint32_t, typename FT=float>
auto knns2graph(const std::vector<packed::pair<FT, IT>> &knns, size_t np, bool mutual=true) {
    MINOCORE_REQUIRE(knns.size() % np == 0, "sanity");
    MINOCORE_REQUIRE(knns.size(), "nonempty");
    unsigned k = knns.size() / np;
    graph::Graph<boost::undirectedS, FT> ret(np);
    for(size_t i = 0; i < np; ++i) {
        auto p = &knns[i * k];
        SK_UNROLL_8
        for(unsigned j = 0; j < k; ++j)
            // This assumes that the distance is symmetric. TODO: correct this assumption.
            if(!mutual || p[j].first <= knns[(p[j].second + 1) * k - 1].first)
                boost::add_edge(i, static_cast<size_t>(p[j].second), p[j].first, ret);
    }
    return ret;
}

template<typename IT=uint32_t, typename MatrixType>
auto make_knn_graph(const jsd::DissimilarityApplicator<MatrixType> &app, unsigned k, bool mutual=true) {
    assert(blz::detail::is_symmetric(app.get_measure()));
    return knns2graph(make_knns(app, k), app.size(), mutual);
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


} // minocore
