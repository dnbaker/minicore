#pragma once
#ifndef FGC_LOCAL_SEARCH_H__
#define FGC_LOCAL_SEARCH_H__
#include "fgc/graph.h"
#include "fgc/diskmat.h"
#include "fgc/kcenter.h"
#include "pdqsort/pdqsort.h"
#include <atomic>

/*
 * In this file, we use the local search heuristic for k-median.
 * Originally described in "Local Search Heuristics for k-median and Facility Location Problems",
 * Vijay Arya, Naveen Garg, Rohit Khandekar, Adam Meyerson, Kamesh Munagala, Vinayaka Pandit
 * (http://theory.stanford.edu/~kamesh/lsearch.pdf)
 */

namespace fgc {
template<typename T> class TD;

template<typename Graph, typename MatType, typename VType=std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>>
void fill_graph_distmat(const Graph &x, MatType &mat, const VType *sources=nullptr, bool only_sources_as_dests=false, bool all_sources=false) {
    const size_t nrows = all_sources || (sources == nullptr) ? boost::num_vertices(x)
                                                             : sources->size();
    if(only_sources_as_dests && sources == nullptr) throw std::invalid_argument("only_sources_as_dests requires sources be non-null");
    const size_t ncol = only_sources_as_dests ? sources->size(): boost::num_vertices(x);
    const typename boost::graph_traits<Graph>::vertex_iterator vertices = boost::vertices(x).first;
    assert(mat.rows() == nrows);
    assert(mat.columns() == ncol);
    if(mat.rows() != nrows || mat.columns() != ncol) {
        char buf[256];
        throw std::invalid_argument(std::string(buf, std::sprintf(buf, "mat sizes (%zu rows, %zu col) don't match output requirements (%zu/%zu)\n",
                                                                  mat.rows(), mat.columns(), nrows, ncol)));
    }
    std::atomic<size_t> rows_complete;
    rows_complete.store(0);
    if(only_sources_as_dests) {
        // require that mat have nrows * ncol
        if(sources == nullptr) throw std::invalid_argument("only_sources_as_dests requires non-null sources");
        unsigned nt = 1;
#if !defined(USE_BOOST_PARALLEL) && defined(_OPENMP)
        OMP_PRAGMA("omp parallel")
        {
            OMP_PRAGMA("omp single")
            nt = omp_get_num_threads();
        }
#endif
        blaze::DynamicMatrix<float> working_space(nt, boost::num_vertices(x));
#ifndef USE_BOOST_PARALLEL
        OMP_PFOR
#endif
        for(size_t i = 0; i < nrows; ++i) {
            unsigned rowid = 0;
#if !defined(USE_BOOST_PARALLEL)
            OMP_ONLY(rowid = omp_get_thread_num();)
#endif
            auto vtx = all_sources ? vertices[i]: (*sources)[i];
            auto wrow(row(working_space, rowid BLAZE_CHECK_DEBUG));
            boost::dijkstra_shortest_paths(x, vtx, distance_map(&wrow[0]));
            row(mat, i BLAZE_CHECK_DEBUG) = blaze::serial(blaze::elements(wrow, sources->data(), sources->size()));
            ++rows_complete;
            const auto val = rows_complete.load();
            if((val & (val - 1)) == 0)
                std::fprintf(stderr, "Completed dijkstra for row %zu/%zu\n", val, nrows);
        }
    } else {
        assert(ncol == boost::num_vertices(x));
#ifndef NDEBUG
        if(all_sources) {
            assert(boost::num_vertices(x) == nrows);
        }
#endif
#ifndef USE_BOOST_PARALLEL
        OMP_PFOR
#endif
        for(size_t i = 0; i < nrows; ++i) {
            auto mr = row(~mat, i BLAZE_CHECK_DEBUG);
            auto vtx = all_sources || sources == nullptr ? vertices[i]: (*sources)[i];
            assert(vtx < boost::num_vertices(x));
            boost::dijkstra_shortest_paths(x, vtx, distance_map(&mr[0]));
            ++rows_complete;
            const auto val = rows_complete.load();
            if((val & (val - 1)) == 0)
                std::fprintf(stderr, "Completed dijkstra for row %zu/%zu\n", val, nrows);
        }
    }
}

template<typename Graph, typename VType=std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>>
DiskMat<typename Graph::edge_property_type::value_type>
graph2diskmat(const Graph &x, std::string path, const VType *sources=nullptr, bool only_sources_as_dests=false, bool all_sources=false) {
    static_assert(std::is_arithmetic<typename Graph::edge_property_type::value_type>::value, "This should be floating point, or at least arithmetic");
    using FT = typename Graph::edge_property_type::value_type;
    size_t nv = sources && only_sources_as_dests ? sources->size(): boost::num_vertices(x);
    size_t nrows = all_sources || !sources ? boost::num_vertices(x): sources->size();
    std::fprintf(stderr, "all sources: %d. nrows: %zu\n", all_sources, nrows);
    DiskMat<FT> ret(nrows, nv, path);
    fill_graph_distmat(x, ret, sources, only_sources_as_dests, all_sources);
    return ret;
}


template<typename Graph, typename VType=std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>>
blz::DynamicMatrix<typename Graph::edge_property_type::value_type>
graph2rammat(const Graph &x, std::string, const VType *sources=nullptr, bool only_sources_as_dests=false, bool all_sources=false) {
    static_assert(std::is_arithmetic<typename Graph::edge_property_type::value_type>::value, "This should be floating point, or at least arithmetic");
    using FT = typename Graph::edge_property_type::value_type;
    size_t nv = sources && only_sources_as_dests ? sources->size(): boost::num_vertices(x);
    size_t nrows = all_sources || !sources ? boost::num_vertices(x): sources->size();
    std::fprintf(stderr, "all sources: %d. nrows: %zu\n", all_sources, nrows);
    blz::DynamicMatrix<FT>  ret(nrows, nv);
    fill_graph_distmat(x, ret, sources, only_sources_as_dests, all_sources);
    return ret;
}

namespace detail {

template<typename T, typename Functor>
void recursive_combinatorial_for_each(size_t nitems, unsigned r, T &ret, size_t npicks, const Functor &func) {
    for(unsigned i = nitems; i >= r; i--) {
        ret[r - 1] = i - 1;
        if(r > 1) {
            recursive_combinatorial_for_each(i - 1, r - 1, ret, npicks, func);
        } else {
            func(ret);
        }
    }
}
#if 0
template<typename IType=uint32_t, size_t N>
struct CombGenerator
{
    using combination_type = blaze::SmallArray<IType, N>;

   CombGenerator(IType N, IType R) :
       completed(N < 1 || R > N),
       generated(0),
       N(N), R(R)
   {
       for (IType c = 0; c < R; ++c)
           current_.pushBack(c);
   }

   // true while there are more solutions
   bool completed;

   // count how many generated
   size_t generated;

   // get current_ent and compute next combination
   combination_type next()
   {
       combination_type ret = current_;

       // find what to increment
       completed = true;
       for (IType i = R - 1; i >= 0; --i)
           if (current_[i] < N - R + i)
           {
               IType j = current_[i];
               while (i <= R-1)
                   current_[i++] = j++;
               completed = false;
               ++generated;
               break;
           }

       return ret;
   }

private:

   const IType N, R;
   combination_type current_;
};
#endif

}

#if 0
std::vector<uint32_t> kcenter_matrix(const Mat &mat, RNG &rng, unsigned k) {
}
                auto approx = kcenter_matrix(mat_, rng, k_);
#endif

template<typename MatType, typename IType=std::uint32_t, size_t N=16>
struct ExhaustiveSearcher {
    using value_type = typename MatType::ElementType;
    const MatType &mat_;
    blaze::SmallArray<IType, N> bestsol_;
    double current_cost_;
    const unsigned k_;
    ExhaustiveSearcher(const MatType &mat, unsigned k): mat_(mat), bestsol_(k_), current_cost_(std::numeric_limits<double>::max()), k_(k) {}
    void run() {
        blaze::SmallArray<IType, N> csol(k_);
        const size_t nr = mat_.rows();
        size_t nchecked = 0;
        detail::recursive_combinatorial_for_each(nr, k_, csol, k_, [&](const auto &sol) {
            const double cost = blz::sum(blz::min<blz::columnwise>(rows(mat_, sol BLAZE_CHECK_DEBUG)));
            ++nchecked;
            if((nchecked & (nchecked - 1)) == 0)
                std::fprintf(stderr, "iteration %zu completed\n", nchecked);
            if(cost < current_cost_) {
                std::fprintf(stderr, "Swapping to new center set with new cost = %g on iteration %zu\n", cost, nchecked);
                current_cost_ = cost;
                bestsol_ = sol;
            }
        });
        std::fprintf(stderr, "Best result: %g. Total number of combinations checked: %zu\n", current_cost_, nchecked);
    }
};

template<typename MatType, typename IType=std::uint32_t>
auto make_kmed_esearcher(const MatType &mat, unsigned k) {
    return ExhaustiveSearcher<MatType, IType>(mat, k);
}

template<typename MatType, typename IType=std::uint32_t>
struct LocalKMedSearcher {
    using value_type = typename MatType::ElementType;


    const MatType &mat_;
    shared::flat_hash_set<IType> sol_;
    blz::DV<IType> assignments_;
    double current_cost_;
    double eps_, initial_cost_, init_cost_div_;
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
    template<typename IndexContainer=std::vector<uint32_t>>
    LocalKMedSearcher(const MatType &mat, unsigned k, double eps=1e-8, uint64_t seed=0,
                      bool best_improvement=false, const IndexContainer *wc=nullptr):
        mat_(mat), assignments_(mat.columns(), 0),
        // center_indices_(k),
        //costs_(mat.columns(), std::numeric_limits<value_type>::max()),
        //counts_(k),
        current_cost_(std::numeric_limits<value_type>::max()),
        eps_(eps),
        k_(k), nr_(mat.rows()), nc_(mat.columns()), best_improvement_(best_improvement)
    {
        static_assert(std::is_integral_v<std::decay_t<decltype(wc->operator[](0))>>, "index container must contain integral values");
        sol_.reserve(k);
        init_cost_div_ = wc
                       ? double(std::accumulate(wc->begin() + 1, wc->end(), wc->operator[](0)))
                       : double(mat.columns());
        reseed(seed, true, wc);
    }

    template<typename IndexContainer=std::vector<uint32_t>>
    void reseed(uint64_t seed, bool do_kcenter=false, const IndexContainer *wc=nullptr) {
        assignments_ = 0;
        current_cost_ = std::numeric_limits<value_type>::max();
        wy::WyRand<IType, 2> rng(seed);
        sol_.clear();
#if 0
        if(wc) {
            // Reweight
            for(unsigned i = 0; i < wc->size(); ++i) {
                column(mat_, i BLAZE_CHECK_DEBUG) *= wc->operator[](col);
            }
        }
#endif
        if(mat_.rows() <= k_) {
            for(unsigned i = 0; i < mat_.rows(); ++i)
                sol_.insert(i);
        } else if(do_kcenter && mat_.rows() == mat_.columns()) {
            std::fprintf(stderr, "Using kcenter\n");
            auto rowits = rowiterator(mat_);
            auto approx = coresets::kcenter_greedy_2approx(rowits.begin(), rowits.end(), rng, k_, MatrixLookup(), std::min(mat_.rows(), mat_.columns()));
            for(const auto c: approx) sol_.insert(c);
#ifndef NDEBUG
            std::fprintf(stderr, "k_: %u. sol size: %zu. rows: %zu. columns: %zu\n", k_, sol_.size(),
                         mat_.rows(), mat_.columns());
#endif
            assert(sol_.size() == k_ || sol_.size() == mat_.rows());
        } else {
            if(!do_kcenter || wc == nullptr || wc->size() != mat_.columns()) {
                while(sol_.size() < k_)
                    sol_.insert(rng() % mat_.rows());
            } else {
                //std::fprintf(stderr, "Using submatrix to perform kcenter approximation on an asymmetric matrix. rows/cols before: %zu, %zu\n", mat_.rows(), mat_.columns());
                blaze::DynamicMatrix<value_type> subm = blaze::rows(mat_, wc->data(), wc->size());
                //std::cerr << subm << '\n';
                //std::fprintf(stderr, "subm rows: %zu\n", subm.rows());
                std::vector<uint32_t> approx{uint32_t(rng() % subm.rows())};
                auto first = approx.front();
                blz::DV<value_type, blaze::rowVector> mincosts = row(subm, first);
                std::vector<uint32_t> remaining(subm.rows());
                std::iota(remaining.begin(), remaining.end(), 0u);
                while(approx.size() < std::min(subm.rows(), size_t(k_))) {
                    //std::fputc('\n', stderr);
                    double maxcost = -1.;
                    unsigned maxind = -1;
                    for(unsigned i = 0; i < remaining.size(); ++i) {
                        auto ri = remaining[i];
                        if(std::find(approx.begin(), approx.end(), ri) != approx.end()) continue;
                        auto r = row(subm, ri);
                        auto cost = blaze::max(r);
                        if(cost > maxcost) maxcost = cost, maxind = i;
                    }
                    auto nextind = remaining[maxind];
                    approx.push_back(nextind);
                    std::swap(remaining[maxind], remaining.back());
                    remaining.pop_back();
                    mincosts = blaze::min(mincosts, row(subm, nextind));
                }
                for(auto i: approx)
                    sol_.insert(wc->at(i));

                while(sol_.size() < k_) {
                    // Add random entries until desired sizeA
                    sol_.insert(rng() % mat_.rows());
                }
                //std::fprintf(stderr, "used submatrix. sol size: %zu\n", sol_.size());
            }
        }
        assign();
    }

    template<typename Container>
    double cost_for_sol(const Container &c) {
        double ret = 0.;
        OMP_PRAGMA("omp parallel for reduction(+:ret)")
        for(size_t i = 0; i < mat_.columns(); ++i) {
            auto col = column(mat_, i);
            auto it = c.begin();
            value_type minv = col[*it];
            while(++it != c.end())
                minv = std::min(col[*it], minv);
            ret += minv;
        }
        return ret;
    }

    // Setup/Utilities

    void assign() {
        assert(assignments_.size() == nc_);
        std::fprintf(stderr, "rows: %zu. cols: %zu. sol size: %zu. k: %u\n",
                     mat_.rows(), mat_.columns(), sol_.size(), k_);
        assert(sol_.size() == k_ || sol_.size() == mat_.rows());
        DBG_ONLY(std::fprintf(stderr, "Initialized assignments at size %zu\n", assignments_.size());)
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
        DBG_ONLY(std::fprintf(stderr, "Set assignments for size %zu\n", assignments_.size());)
        current_cost_ = cost_for_sol(sol_);
        DBG_ONLY(std::fprintf(stderr, "Got costs for size %zu with centers size = %zu\n", assignments_.size(), sol_.size());)
        initial_cost_ = current_cost_ / 2 / init_cost_div_;
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
#define LOOP_CORE(ind) \
        do {\
            const auto asn = assignments_[ind];\
            assert(asn < nr_);\
            const auto old_cost = mat_(asn, ind);\
            if(asn == oldcenter) {\
                value_type newcost = newr[ind];\
                for(const auto ctr: sol_) {\
                    if(ctr != oldcenter)\
                        newcost = std::min(mat_(ctr, ind), newcost);\
                }\
                potential_gain += old_cost - newcost;\
            } else if(double nv = newr[ind]; nv < old_cost) {\
                potential_gain += old_cost - nv;\
            }\
        } while(0)

        if(single_threaded) {
            for(size_t i = 0; i < nc_; ++i) {
                LOOP_CORE(i);
            }
        } else {
            OMP_PRAGMA("omp parallel for reduction(+:potential_gain)")
            for(size_t i = 0; i < nc_; ++i) {
                LOOP_CORE(i);
            }
        }
#undef LOOP_CORE
        return potential_gain;
    }

    // Getters
    auto k() const {
        return k_;
    }

    bool recalculate() {
        assignments_ = *sol_.begin();

        for(auto it = sol_.begin(); ++it != sol_.end();) {
            const auto center = *it;
            auto crow = row(mat_, center BLAZE_CHECK_DEBUG);
            assert(crow.size() == nc_);
            OMP_PFOR
            for(size_t i = 0; i < nc_; ++i) {
                if(const auto cost(crow[i]); cost < mat_(assignments_[i], i)) {
                    assignments_[i] = center;
                }
            }
        }

        value_type newcost = 0.;
        OMP_PRAGMA("omp parallel for reduction(+:newcost)")
        for(size_t i = 0; i < assignments_.size(); ++i) {
                newcost += mat_(assignments_[i], i);
        }
        //std::fprintf(stderr, "newcost: %f. old cost: %f\n", newcost, current_cost_);
        if(unlikely(newcost > current_cost_)) {
            std::fprintf(stderr, "Somehow this swap is bad. newcost: %g. old: %g. diff: %g\n", newcost, current_cost_, current_cost_ - newcost);
            return true;
        }
        assert(newcost <= current_cost_ || !std::fprintf(stderr, "newcost: %g. old: %g. diff: %g\n", newcost, current_cost_, current_cost_ - newcost));
        current_cost_ = newcost;
        return false;
    }

    void run() {
        if(mat_.rows() < k_) return;
        //const double diffthresh = 0.;
        const double diffthresh = initial_cost_ / k_ * eps_;
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
                                    current_best_center = oldcenter;
                                }
                                std::fprintf(stderr, "Swapping %zu for %u. Swap number %zu. Current cost: %g. Improvement: %g\n", pi, oldcenter, total + 1, current_cost_, val);
                            }
                        }
                    }
                }
                if(current_best_index != IType(-1)) {
                    sol_.erase(current_best_center);
                    sol_.insert(current_best_index);
                    if(!recalculate()) {
                        ++total;
                        goto next;
                    }
                    sol_.insert(current_best_center);
                    sol_.erase(current_best_index);
                    //diffthresh = current_cost_ / k_ * eps_;
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
                                if(recalculate()) {
                                    sol_.insert(oldcenter);
                                    sol_.erase(pi);
                                    continue;
                                }
                                ++total;
                                goto next;
                            }
                        }
                    }
                }
                // The solution is usually extremely close even if there is a change here. (less than 1e-6 of the cost)
#ifndef NDEBUG
                exhaustive_manual_check();
#endif
            }
        }
        std::fprintf(stderr, "Finished in %zu swaps by exhausting all potential improvements. Final cost: %f\n",
                     total, current_cost_);
    }
    void exhaustive_manual_check() {
        const std::vector<IType> csol(sol_.begin(), sol_.end());
        std::vector<IType> wsol = csol, fsol = csol;
        double ccost = current_cost_;
#ifndef NDEBUG
        double ocost = current_cost_;
#endif
        size_t extra_rounds = 0;
        bool improvement_made;
        start:
        improvement_made = false;
        for(size_t si = 0; si < k_; ++si) {
            for(size_t ci = 0; ci < nr_; ++ci) {
                if(std::find(wsol.begin(), wsol.end(), ci) != wsol.end()) continue;
                wsol[si] = ci;
                const double cost = blz::sum(blz::min<blz::columnwise>(rows(mat_, wsol)));
                if(cost < ccost) {
                    std::fprintf(stderr, "Found a better one: %g vs %g (%g)\n", cost, ccost, ccost - cost);
                    ccost = cost;
                    fsol = wsol;
                    wsol = fsol;
                    improvement_made = true;
                    ++extra_rounds;
                    goto start;
                }
            }
            wsol[si] = csol[si];
        }
        if(improvement_made) goto start;
        current_cost_ = ccost;
#ifndef NDEBUG
        std::fprintf(stderr, "improved cost for %zu rounds and a total improvemnet of %g\n", extra_rounds, ocost - current_cost_);
        //assert(std::abs(ocost - current_cost_) < ((initial_cost_ / k_ * eps_) + 0.1));  // 1e-5 for numeric stability issues
#endif
    }
};

template<typename Mat, typename IType=std::uint32_t, typename IndexContainer=std::vector<uint32_t>>
auto make_kmed_lsearcher(const Mat &mat, unsigned k, double eps=0.01, uint64_t seed=0, bool best_improvement=false,
                         const IndexContainer *wc=nullptr) {
    return LocalKMedSearcher<Mat, IType>(mat, k, eps, seed, best_improvement, wc);
}


} // fgc

#endif /* FGC_LOCAL_SEARCH_H__ */
