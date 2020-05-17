#include "minocore/graph.h"
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
    const DissimilarityMeasure measure = app.get_measure();
    std::vector<packed::pair<FT, IT>> ret(k * np, static_cast<IT>(-1));
    std::vector<unsigned> in_set(np);
    const bool measure_is_sym = detail::is_symmetric(measure);
    const bool measure_is_dist = measure_is_dist;
    std::unique_ptr<std::mutex[]> locks;
    OMP_ONLY(locks.reset(new std::mutex[np]);)

    auto update_fwd = [&](FT d, size_t i, size_t j) {
        if(in_set[i] < k) {
            std::lock_guard<std::mutex> lock(locks[i]);
            ret[i * k + in_set[i]] = packed::pair<FT, IT>{d, j};
            if(++in_set[i] == k) {
                if(measure_is_dist)
                    std::make_heap(ret.data() + i * k, ret.data() + (i + 1) * k, std::less<void>());
                else
                    std::make_heap(ret.data() + i * k, ret.data() + (i + 1) * k, std::greater<void>());
            }
        } else {
            if(measure_is_dist) {
                if(ret[i * k].first > d) {
                    std::lock_guard<std::mutex> lock(locks[i]);
                    if(ret[i * k].first > d) {
                        std::pop_heap(&ret[i * k], &ret[(i + 1) * k], std::less<void>());
                        ret[(i + 1) * k - 1] = packed::pair<FT, IT>{d, j};
                        std::push_heap(&ret[i * k], &ret[(i + 1) * k], std::less<void>());
                    }
                } 
            } else {
                if(ret[i * k].first < d) {
                    std::lock_guard<std::mutex> lock(locks[i]);
                    if(ret[i * k].first < d) {
                        std::pop_heap(&ret[i * k], &ret[(i + 1) * k], std::greater<void>());
                        ret[(i + 1) * k - 1] = packed::pair<FT, IT>{d, j};
                        std::push_heap(&ret[i * k], &ret[(i + 1) * k], std::less<void>());
                    }
                } 
            }
        }
    };

    // Helper functions
    if(measure_is_sym) {
        OMP_PFOR
        for(size_t i = 0; i < np; ++i) {
            for(size_t j = i + 1; j < np; ++j) {
                auto d = app(i, j);
                update_fwd(d, i, j);
                update_fwd(d, j, i);
            }
        }
    } else {
        OMP_PFOR
        for(size_t i = 0; i < np; ++i) {
            for(size_t j = 0; j < np; ++j) {
                update_fwd(app(i, j), i, j);
            }
        }
    }
    OMP_PFOR
    for(size_t i = 0; i < np; ++i) {
        auto start = ret.data() + i * k, end = start + k;
        if(measure_is_dist)
            shared::sort(start, end, std::less<>());
        else
            shared::sort(start, end, std::greater<>());
    }
    return ret;
}

template<typename IT=uint32_t, typename FT=float>
auto knns2graph(const std::vector<packed::pair<FT, IT>> &knns, size_t np, bool mutual=true) {
    MINOCORE_REQUIRE(knns.size() % np == 0, "sanity");
    MINOCORE_REQUIRE(knns.size(), "nonempty");
    unsigned k = knns.size() / np;
    graph::Graph<FT> ret(np);
    if(mutual) {
        OMP_PFOR
        for(size_t i = 0; i < np; ++i)
            mxds[i] = knns[(i + 1) * k - 1].first;
        for(size_t i = 0; i < np; ++i) {
            auto p = &knns[i * k];
            SK_UNROLL_8
            for(unsigned j = 0; i < k; ++j) {
                FT d = p[j].first;
                IT ind = p[j].second;
                if(d <= mxds[ind])
                    boost::add_edge(i, ind, d, ret);
            }
        }
    } else {
        for(size_t i = 0; i < np; ++i) {
            auto p = &knns[i * k];
            SK_UNROLL_4
            for(unsigned j = 0; i < k; ++j)
                boost::add_edge(i, p[j].second, p[j].first, ret);
        }
    }
    return ret;
}


} // minocore
