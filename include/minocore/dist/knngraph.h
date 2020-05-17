#include "minocore/graph.h"
#include "minocore/util/packed.h"
#include "minocore/dist/applicator.h"

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
            if(!mutual || p[j].first <= knns[(p[j].second + 1) * k - 1].first)
                boost::add_edge(i, static_cast<size_t>(p[j].second), p[j].first, ret);
    }
    return ret;
}

template<typename IT=uint32_t, typename MatrixType>
auto make_knn_graph(const jsd::DissimilarityApplicator<MatrixType> &app, unsigned k, bool mutual=true) {
    return knns2graph(make_knns(app, k), app.size(), mutual);
}


} // minocore
