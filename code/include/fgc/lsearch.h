#pragma once
#ifndef FGC_LOCAL_SEARCH_H__
#define FGC_LOCAL_SEARCH_H__
#include "fgc/graph.h"
#include "fgc/diskmat.h"
#include "fgc/kcenter.h"
#include "pdqsort/pdqsort.h"

/*
 * In this file, we use the local search heuristic for k-median.
 * Originally described in "Local Search Heuristics for k-median and Facility Location Problems",
 * Vijay Arya, Naveen Garg, Rohit Khandekar, Adam Meyerson, Kamesh Munagala, Vinayaka Pandit
 * (http://theory.stanford.edu/~kamesh/lsearch.pdf)
 */

namespace fgc {

template<typename Graph, typename MatType, typename VType=std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>>
void fill_graph_distmat(const Graph &x, MatType &mat, const VType *sources=nullptr, bool sources_only=false) {
    const size_t nrows = sources ? sources->size(): boost::num_vertices(x);
    size_t ncol = sources_only ? nrows: boost::num_vertices(x);
    const typename boost::graph_traits<Graph>::vertex_iterator vertices = boost::vertices(x).first;
    if(mat.rows() != nrows || mat.columns() != ncol) throw std::invalid_argument("mat sizes don't match output");
#ifndef NDEBUG
    auto vtx_idx_map = boost::get(vertex_index, x);
#endif
    if(sources_only) {
        // require that mat have nrows * ncol
        if(sources == nullptr) throw std::invalid_argument("sources_only requires non-null sources");
        unsigned nt = 1;
#ifdef _OPENMP
        OMP_PRAGMA("omp parallel")
        {
            OMP_PRAGMA("omp single")
            nt = omp_get_num_threads();
        }
#endif
        blaze::DynamicMatrix<float> working_space(nt, boost::num_vertices(x));
        OMP_PFOR
        for(size_t i = 0; i < nrows; ++i) {
            unsigned rowid = 0;
#ifdef _OPENMP
            rowid = omp_get_thread_num();
#endif
            auto wrow(row(working_space, rowid BLAZE_CHECK_DEBUG));
            auto vtx = (*sources)[i];
            boost::dijkstra_shortest_paths(x, vtx, distance_map(&wrow[0]));
            //std::fprintf(stderr, "Calculated dijkstra for row %zu from thread %u\n", i, rowid);
            row(mat, i BLAZE_CHECK_DEBUG) = blaze::serial(blaze::elements(wrow, sources->data(), sources->size()));
            //std::fprintf(stderr, "Assigned row %zu from thread %u\n", i, rowid);
        }
    } else {
        OMP_PFOR
        for(size_t i = 0; i < nrows; ++i) {
            auto mr = row(~mat, i BLAZE_CHECK_DEBUG);
            auto vtx = sources ? (*sources)[i]: vertices[i];
            assert(vtx == vtx_idx_map[vtx]);
            assert(vtx < boost::num_vertices(x));
            boost::dijkstra_shortest_paths(x, vtx, distance_map(&mr[0]));
        }
    }
}

template<typename Graph, typename VType=std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>>
DiskMat<typename Graph::edge_property_type::value_type>
graph2diskmat(const Graph &x, std::string path, const VType *sources=nullptr, bool sources_only=false) {
    static_assert(std::is_arithmetic<typename Graph::edge_property_type::value_type>::value, "This should be floating point, or at least arithmetic");
    using FT = typename Graph::edge_property_type::value_type;
    size_t nv = sources && sources_only ? sources->size(): boost::num_vertices(x), nrows = sources ? sources->size(): nv;
    DiskMat<FT> ret(nrows, nv, path);
    fill_graph_distmat(x, ret, sources, sources_only);
    return ret;
}


template<typename Graph, typename VType=std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>>
blz::DynamicMatrix<typename Graph::edge_property_type::value_type>
graph2rammat(const Graph &x, std::string, const VType *sources=nullptr, bool sources_only=false) {
    static_assert(std::is_arithmetic<typename Graph::edge_property_type::value_type>::value, "This should be floating point, or at least arithmetic");
    using FT = typename Graph::edge_property_type::value_type;
    size_t nv = sources && sources_only ? sources->size(): boost::num_vertices(x), nrows = sources ? sources->size(): nv;
    blz::DynamicMatrix<FT>  ret(nrows, nv);
    fill_graph_distmat(x, ret, sources, sources_only);
    return ret;
}


template<typename MatType, typename WFT=float, typename IType=std::uint32_t>
struct LocalKMedSearcher {
    using SolType = blaze::SmallArray<IType, 16>;
    using value_type = typename MatType::ElementType;


    const MatType &mat_;
    shared::flat_hash_set<IType> sol_;
    blz::DV<IType> assignments_;
    double current_cost_;
    double eps_, initial_cost_;
    IType k_;
    const size_t nr_, nc_;
    bool best_improvement_;

    // Constructors

    LocalKMedSearcher(const LocalKMedSearcher &o) = default;
    LocalKMedSearcher(LocalKMedSearcher &&o) {
        auto ptr = reinterpret_cast<const uint8_t *>(this);
        std::memset(ptr, 0, sizeof(*this));
        std::swap_ranges(ptr, ptr + sizeof(*this), reinterpret_cast<const uint8_t *>(std::addressof(o)));
    }
    LocalKMedSearcher(const MatType &mat, unsigned k, double eps=0.01, uint64_t seed=0, bool best_improvement=false):
        mat_(mat), assignments_(mat.columns(), 0),
        // center_indices_(k),
        //costs_(mat.columns(), std::numeric_limits<value_type>::max()),
        //counts_(k),
        current_cost_(std::numeric_limits<value_type>::max()),
        eps_(eps),
        k_(k), nr_(mat.rows()), nc_(mat.columns()), best_improvement_(best_improvement)
    {
        sol_.reserve(k);
        reseed(seed, true);
    }

    void reseed(uint64_t seed, bool do_kcenter=false) {
        assignments_ = 0;
        current_cost_ = std::numeric_limits<value_type>::max();
        wy::WyRand<IType, 2> rng(seed);
        sol_.clear();
        if(do_kcenter) {
            auto rowits = rowiterator(mat_);
            auto approx = coresets::kcenter_greedy_2approx(rowits.begin(), rowits.end(), rng, k_, MatrixLookup());
            for(const auto c: approx) sol_.insert(c);
        } else {
            while(sol_.size() < k_)
                sol_.insert(rng() % mat_.rows());
        }
        assign();
        initial_cost_ = current_cost_ / 2 / nc_;
    }

    template<typename Container>
    double cost_for_sol(const Container &c) {
        double ret = 0.;
        OMP_PRAGMA("omp parallel for reduction(+:ret)")
        for(size_t i = 0; i < mat_.columns(); ++i) {
            auto col = column(mat_, i);
            auto it = c.begin();
            value_type minv = col[*it++];
            while(it != c.end())
                minv = std::min(col[*it++], minv);
            ret += minv;
        }
        return ret;
    }

    // Setup/Utilities

    void assign() {
        assert(assignments_.size() == nc_);
        assert(sol_.size() == k_);
        for(const auto center: sol_) {
            assignments_[center] = center;
            //assert(mat_(center, center) == 0.);
        }
        for(const auto center: sol_) {
            auto r = row(mat_, center BLAZE_CHECK_DEBUG);
            OMP_PFOR
            for(size_t ci = 0; ci < nc_; ++ci) {
                if(const auto newcost = r[ci];
                   newcost < mat_(assignments_[ci], ci))
                {
                   assignments_[ci] = center;
                }
            }
        }
        current_cost_ = cost_for_sol(sol_);
    }

    double evaluate_swap(IType newcenter, IType oldcenter, bool single_threaded=false) const {
        //std::fprintf(stderr, "[%s] function starting: %u/%u\n", __PRETTY_FUNCTION__, newcenter, oldcenter);
        assert(newcenter < mat_.rows());
        assert(oldcenter < mat_.rows());
        //std::fprintf(stderr, "Passed asserts\n");
        auto newr = row(mat_, newcenter BLAZE_CHECK_DEBUG);
        //std::fprintf(stderr, "Got row: %zu\n", newr.size());
        assert(nc_ == newr.size());
        assert(assignments_.size() == mat_.columns());
        double potential_gain = 0.;
        auto eval_func = [&,oc=oldcenter](size_t ind) {
            const auto asn = assignments_[ind];
            assert(asn < nr_);
            const auto old_cost = mat_(asn, ind);
            value_type nv = newr[ind];
            if(asn == oc) {
                for(const auto ctr: sol_) {
                    if(ctr != oc)
                        nv = std::min(mat_(ctr, ind), nv);
                }
                potential_gain += old_cost - newcost;
            } else if(nv < old_cost) {
                potential_gain += old_cost - nv;
            }
        };
        if(single_threaded) {
            for(size_t i = 0; i < nc_; ++i) {
                eval_func(i);
            }
        } else {
            OMP_PRAGMA("omp parallel for reduction(+:potential_gain)")
            for(size_t i = 0; i < nc_; ++i) {
                eval_func(i);
            }
        }
        return potential_gain;
    }

    // Getters
    auto k() const {
        return k_;
    }

    void recalculate() {
        assignments_ = *sol_.begin();

        for(auto it = sol_.begin(); ++it != sol_.end();) {
            const auto center = *it;
            auto crow = row(mat_, center BLAZE_CHECK_DEBUG);
            assert(crow.size() == nc_);
            OMP_PFOR
            for(size_t i = 0; i < nc_; ++i) {
                if(const auto cost(crow[i]); cost < mat_(assignments_[i], i))
                    assignments_[i] = center;
            }
        }
#ifndef NDEBUG
        for(const auto asn: assignments_)
            assert(std::find(sol_.begin(), sol_.end(), asn) != sol_.end());
#endif
        double newcost = 0.;
        OMP_PRAGMA("omp parallel for reduction(+:newcost)")
        for(size_t i = 0; i < nc_; ++i)
            newcost += mat_(assignments_[i], i);


        //std::fprintf(stderr, "newcost: %f. old cost: %f\n", newcost, current_cost_);
        assert(newcost <= current_cost_);
        current_cost_ = newcost;
    }

    void run() {
        double diffthresh = initial_cost_ / k_ * eps_;
        std::fprintf(stderr, "diffthresh: %f\n", diffthresh);
        size_t total = 0;
        {
            double current_best;
            IType current_best_index, current_best_center;
            next:
            if(best_improvement_) {
                current_best = std::numeric_limits<double>::min();
                current_best_index = -1, current_best_center = -1;
                for(const auto oldcenter: sol_) {
                    OMP_PFOR
                    for(size_t pi = 0; pi < nr_; ++pi) {
                        if(sol_.find(pi) == sol_.end()) {
                            if(const auto val = evaluate_swap(pi, oldcenter, true);
                               val > diffthresh && val > current_best)
                            {
                                OMP_CRITICAL
                                {
                                    current_best = val;
                                    current_best_index = pi;
                                }
                                current_best_center = oldcenter;
                                std::fprintf(stderr, "Swapping %zu for %u. Swap number %zu. Current cost: %g. Improvement: %g\n", pi, oldcenter, total + 1, current_cost_, val);
                            }
                        }
                    }
                }
                if(current_best_index != IType(-1)) {
                    sol_.erase(current_best_center);
                    sol_.insert(current_best_index);
                    recalculate();
                    //diffthresh = current_cost_ / k_ * eps_;
                    ++total;
                    goto next;
                }
            } else {
                for(const auto oldcenter: sol_) {
                    for(size_t pi = 0; pi < nr_; ++pi) {
                        if(sol_.find(pi) == sol_.end()) {
                            if(const auto val = evaluate_swap(pi, oldcenter);
                               val > diffthresh) {
                                std::fprintf(stderr, "Swapping %zu for %u. Swap number %zu. Current cost: %g. Improvement: %g. Threshold: %g.\n", pi, oldcenter, total + 1, current_cost_, val, diffthresh);
                                sol_.erase(oldcenter);
                                sol_.insert(pi);
                                recalculate();
                                //diffthresh = current_cost_ / k_ * eps_;
                                ++total;
                                goto next; // Meaning we've swapped this guy out and will pick another one.
                            }
                        }
                    }
                }
            }
        }
        std::fprintf(stderr, "Finished in %zu swaps by exhausting all potential improvements. Final cost: %f\n",
                     total, current_cost_);
    }
};

template<typename Mat, typename FT=float, typename IType=std::uint32_t>
auto make_kmed_lsearcher(const Mat &mat, unsigned k, double eps=0.01, uint64_t seed=0, bool best_improvement=false) {
    return LocalKMedSearcher<Mat, FT, IType>(mat, k, eps, seed, best_improvement);
}


} // fgc

#endif /* FGC_LOCAL_SEARCH_H__ */
