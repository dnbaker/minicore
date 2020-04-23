#pragma once
#ifndef FGC_LOCAL_SEARCH_H__
#define FGC_LOCAL_SEARCH_H__
#include "minocore/util/diskmat.h"
#include "minocore/util/oracle.h"
#include "minocore/optim/kcenter.h"
#include "pdqsort/pdqsort.h"
#include "discreture/include/discreture.hpp"
#include <atomic>

/*
 * In this file, we use the local search heuristic for k-median.
 * Originally described in "Local Search Heuristics for k-median and Facility Location Problems",
 * Vijay Arya, Naveen Garg, Rohit Khandekar, Adam Meyerson, Kamesh Munagala, Vinayaka Pandit
 * (http://theory.stanford.edu/~kamesh/lsearch.pdf)
 */

namespace minocore {

namespace graph {


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
        for(auto &&comb: discreture::combinations(nr, k_)) {
            const double cost = blz::sum(blz::min<blz::columnwise>(rows(mat_, comb.data(), comb.size())));
            ++nchecked;
            if((nchecked & (nchecked - 1)) == 0)
                std::fprintf(stderr, "iteration %zu completed\n", nchecked);
            if(cost < current_cost_) {
                std::fprintf(stderr, "Swapping to new center set with new cost = %g on iteration %zu\n", cost, nchecked);
                current_cost_ = cost;
                std::copy(comb.data(), comb.data() + comb.size(), bestsol_.data());
            }
        }
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
    blz::DV<typename MatType::ElementType, blaze::rowVector> current_costs_;
    double current_cost_;
    double eps_, initial_cost_, init_cost_div_;
    IType k_;
    const size_t nr_, nc_;
    double diffthresh_;
    blz::DV<IType> ordering_;
    uint32_t shuffle_:1;
    // Set to 0 to avoid lazy search, 1 to only do local search, and 2 to do lazy search and then use exhaustive
    uint32_t lazy_eval_:15;
    uint32_t max_swap_n_:16;
    // if(max_swap_n_ > 1), after exhaustive single-swap optimization, enables multiswap search.
    // TODO: enable searches for multiswaps.

    // Constructors

    LocalKMedSearcher(const LocalKMedSearcher &o) = default;
    LocalKMedSearcher(LocalKMedSearcher &&o) {
        auto ptr = reinterpret_cast<const uint8_t *>(this);
        std::memset(ptr, 0, sizeof(*this));
        std::swap_ranges(ptr, ptr + sizeof(*this), reinterpret_cast<const uint8_t *>(std::addressof(o)));
    }
    template<typename IndexContainer=std::vector<uint32_t>>
    LocalKMedSearcher(const MatType &mat, unsigned k, double eps=1e-8, uint64_t seed=0,
                      const IndexContainer *wc=nullptr, double initdiv=0.):
        mat_(mat), assignments_(mat.columns(), 0),
        // center_indices_(k),
        //costs_(mat.columns(), std::numeric_limits<value_type>::max()),
        //counts_(k),
        current_cost_(std::numeric_limits<value_type>::max()),
        eps_(eps),
        k_(k), nr_(mat.rows()), nc_(mat.columns()),
        ordering_(mat.rows()), shuffle_(true), lazy_eval_(1), max_swap_n_(1)
    {
        std::iota(ordering_.begin(), ordering_.end(), 0);
        static_assert(std::is_integral_v<std::decay_t<decltype(wc->operator[](0))>>, "index container must contain integral values");
        sol_.reserve(k);
        init_cost_div_ = initdiv ? initdiv: double(mat.columns());
        reseed(seed, true, wc);
    }

    template<typename It>
    void assign_centers(It start, It end) {
        sol_.clear();
        sol_.insert(start, end);
        assignments_ = 0;
    }

    template<typename IndexContainer=std::vector<uint32_t>>
    void reseed(uint64_t seed, bool do_kcenter=false, const IndexContainer *wc=nullptr) {
        assignments_ = 0;
        current_cost_ = std::numeric_limits<value_type>::max();
        wy::WyRand<IType, 2> rng(seed);
        sol_.clear();
        if(mat_.rows() <= k_) {
            for(unsigned i = 0; i < mat_.rows(); ++i)
                sol_.insert(i);
        } else if(do_kcenter && mat_.rows() == mat_.columns()) {
            std::fprintf(stderr, "Using kcenter\n");
            auto rowits = blz::rowiterator(mat_);
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
    }

    template<typename Container>
    double cost_for_sol(const Container &c) const {
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
        auto it = sol_.begin();
        const auto eit = sol_.end();
        assignments_ = *it;
        current_costs_ = row(mat_, *it);
        while(++it != eit) {
            auto center = *it;
            auto r = row(mat_, center BLAZE_CHECK_DEBUG);
            OMP_PFOR
            for(size_t ci = 0; ci < nc_; ++ci) {
                auto asn = assignments_[ci];
                if(const auto newcost = r[ci];
                   newcost < mat_(asn, ci))
                {
                    current_costs_[ci] = newcost;
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
        blz::SmallArray<IType, 16> as(sol_.begin(), sol_.end());
        *std::find(as.begin(), as.end(), oldcenter) = newcenter;
        double cost;
        if(single_threaded) {
            cost = blaze::serial(blz::sum(blz::serial(blz::min<blz::columnwise>(rows(mat_, as)))));
        } else cost = blz::sum(blz::min<blz::columnwise>(rows(mat_, as)));
        return current_cost_ - cost;
    }

    template<size_t N, typename IndexType>
    double evaluate_multiswap(const IndexType *newcenter, const IndexType *oldcenter, bool single_threaded=false) const {
        blz::SmallArray<IType, 16> as(sol_.begin(), sol_.end());
        shared::sort(as.begin(), as.end());
        for(size_t i = 0; i < N; ++i) {
            *std::find(as.begin(), as.end(), oldcenter[i]) = newcenter[i];
        }
        double cost;
        if(single_threaded) {
            cost = blaze::serial(blz::sum(blz::serial(blz::min<blz::columnwise>(rows(mat_, as)))));
        } else
            cost = blz::sum(blz::min<blz::columnwise>(rows(mat_, as)));
        return current_cost_ - cost;
    }
    template<typename IndexType>
    double evaluate_multiswap_rt(const IndexType *newcenter, const IndexType *oldcenter, size_t N, bool single_threaded=false) const {
        switch(N) {
           case 2: return evaluate_multiswap<2>(newcenter, oldcenter, single_threaded);
           case 3: return evaluate_multiswap<3>(newcenter, oldcenter, single_threaded);
        }
        blz::SmallArray<IType, 16> as(sol_.begin(), sol_.end());
        for(size_t i = 0; i < N; ++i) {
            *std::find(as.begin(), as.end(), oldcenter[i]) = newcenter[i];
        }
        shared::sort(as.begin(), as.end());
        double cost;
        if(single_threaded) {
            cost = blaze::serial(blz::sum(blz::serial(blz::min<blz::columnwise>(rows(mat_, as)))));
        } else
            cost = blz::sum(blz::min<blz::columnwise>(rows(mat_, as)));
        return current_cost_ - cost;
    }

    template<size_t N>
    double lazy_evaluate_multiswap(const IType *newcenters, const IType *oldcenters) const {
        // Instead of performing the full recalculation, do lazy calculation.
        //
        std::vector<IType> tmp(sol_.begin(), sol_.end());
        for(unsigned i = 0; i < N; ++i)
            tmp.erase(std::find(tmp.begin(), tmp.end(), oldcenters[i]));
        std::sort(tmp.begin(), tmp.end());
        // Instead of performing the full recalculation, do lazy calculation.
        if(current_costs_.size() != nc_) { // If not calculated, calculate
            auto it = sol_.begin();
            OMP_CRITICAL
            {
                current_costs_ = row(mat_, *it BLAZE_CHECK_DEBUG);
            }
            while(++it != sol_.end()) {
                current_costs_ = blz::min(current_costs_, row(mat_, *it BLAZE_CHECK_DEBUG));
            }
        }
        blz::DV<typename MatType::ElementType, blz::rowVector> newptr = blz::min<blz::rowwise>(rows(mat_, newcenters, N));
        blz::DV<typename MatType::ElementType, blz::rowVector> oldptr = blz::min<blz::rowwise>(rows(mat_, oldcenters, N));
        double diff = 0.;
#ifdef _OPENMP
        _Pragma("omp parallel for reduction(+:diff)")
#endif
        for(size_t i = 0; i < nc_; ++i) {
            auto ccost = current_costs_[i];
            if(newptr[i] < ccost) {
                auto sub = ccost - newptr[i];
                diff += sub;
            } else if(ccost == oldptr[i]) {
                auto oldbest = blz::min(blz::elements(blz::column(mat_, i), tmp.data(), tmp.size()));
                auto sub = ccost - std::min(oldbest, newptr[i]);
                diff += sub;
            }
        }
        return diff;
    }

    // Getters
    auto k() const {
        return k_;
    }

    void run_lazy() {
#if 0
        shared::flat_hash_map<IType, std::vector<IType>> current_assignments;
        for(size_t i = 0; i < assignments_.size(); ++i) {
            current_assignments[assignments_[i]].push_back(i);
        }
#endif
        size_t total = 0;
        std::vector<IType> newindices(sol_.begin(), sol_.end());
        next:
        for(const auto oldcenter: sol_) {
            std::swap(*std::find(newindices.begin(), newindices.end(), oldcenter), newindices.back());
            if(shuffle_) {
                wy::WyRand<uint64_t, 2> rng(total);
                std::shuffle(ordering_.begin(), ordering_.end(), rng);
            }
            // Make a vector with the original solution, but replace the old value with the new value
            for(size_t pi = 0; pi < nr_; ++pi) {
                auto potential_index = ordering_[pi];
                if(sol_.find(potential_index) != sol_.end()) continue;
                newindices.back() = potential_index;
                assert(std::find(newindices.begin(), newindices.end(), oldcenter) == newindices.end());
                double val = 0.;
                auto newptr = row(mat_, potential_index);
#ifdef _OPENMP
                #pragma omp parallel for reduction(+:val)
#endif
                for(size_t i = 0; i < nc_; ++i) {
                    auto oldcost = current_costs_[i];
                    if(newptr[i] < oldcost) {
                        auto diff = oldcost - newptr[i];
                        val += diff;
                    } else if(assignments_[i] == oldcenter) {
                        auto mincost = blz::min(blz::elements(blz::column(mat_, i), newindices.data(), newindices.size()));
                        auto diff = oldcost - mincost;
                        val += diff;
                    }
                }
#ifndef NDEBUG
                //auto v = evaluate_swap(potential_index, oldcenter);
                //assert(std::abs(v - val) <= .5 * std::abs(std::max(v, val)) || !std::fprintf(stderr, "Manual: %g. Lazy: %g\n", v, val));
                assert(sol_.size() == k_);
#endif
                // Only calculate exhaustively if the lazy form returns yes.
                if(val > diffthresh_ && (val = evaluate_swap(potential_index, oldcenter) > diffthresh_)) {
                    assert(sol_.size() == k_);
                    sol_.erase(oldcenter);
                    sol_.insert(potential_index);
                    assert(sol_.size() == k_);
                    assign();
                    //current_cost_ = blz::sum(current_costs_);
                    ++total;
                    std::fprintf(stderr, "Swap number %zu updated with delta %g to new cost with cost %0.12g\n", total, val, current_cost_);
                    goto next;
                }
            }
        }
        std::fprintf(stderr, "Finished in %zu swaps by exhausting all potential improvements. Final cost: %f\n",
                     total, current_cost_);
    }

    void run_multi(unsigned nswap=1) {
        if(mat_.rows() <= k_) return;
        if(nswap == 1) {
            run();
            return;
        }
        if(nswap >= k_) throw std::runtime_error("nswap >= k_");
        assign();
        const double diffthresh = initial_cost_ / k_ * eps_;
        diffthresh_ = diffthresh;
        next:
        {
            blz::DV<IType> csol(sol_.size());
            std::copy(sol_.begin(), sol_.end(), csol.data());
            blz::DV<IType> swap_in(nc_ - sol_.size());
            blz::DV<IType> inargs(nswap), outargs(nswap);
            for(auto &&swap_out_comb: discreture::combinations(csol.size(), nswap)) {
                for(auto &&swap_in_comb: discreture::combinations(swap_in.size(), nswap)) {
                    auto v = evaluate_multiswap_rt(swap_in_comb.data(), swap_out_comb.data(), nswap);
                    if(v >= diffthresh_) {
                        for(auto v: swap_out_comb) sol_.erase(v);
                        sol_.insert(swap_in_comb.begin(), swap_in_comb.end());
                        current_cost_ -= v;
                        goto next;
                    }
                }
            }
        }
    }
    void run() {
        assign();
        const double diffthresh = initial_cost_ / k_ * eps_;
        diffthresh_ = diffthresh;
        if(mat_.rows() <= k_) return;
        if(lazy_eval_) {
            run_lazy();
            if(lazy_eval_ > 1)
                return;
        }
        //const double diffthresh = 0.;
        std::fprintf(stderr, "diffthresh: %f\n", diffthresh);
        size_t total = 0;
        next:
        for(const auto oldcenter: sol_) {
            if(shuffle_) {
                wy::WyRand<uint64_t, 2> rng(total);
                std::shuffle(ordering_.begin(), ordering_.end(), rng);
            }
            std::vector<IType> newindices(sol_.begin(), sol_.end());
            for(size_t pi = 0; pi < nr_; ++pi) {
                size_t potential_index = ordering_[pi];
                if(sol_.find(potential_index) != sol_.end()) continue;
                if(const auto val = evaluate_swap(potential_index, oldcenter, true);
                   val > diffthresh) {
#ifndef NDEBUG
                    std::fprintf(stderr, "Swapping %zu for %u. Swap number %zu. Current cost: %g. Improvement: %g. Threshold: %g.\n", potential_index, oldcenter, total + 1, current_cost_, val, diffthresh);
#endif
                    sol_.erase(oldcenter);
                    sol_.insert(potential_index);
                    ++total;
                    current_cost_ -= val;
                    std::fprintf(stderr, "Swap number %zu with cost %0.12g\n", total, current_cost_);
                    goto next;
                }
            }
       }
        std::fprintf(stderr, "Finished in %zu swaps by exhausting all potential improvements. Final cost: %f\n",
                     total, current_cost_);
        if(max_swap_n_ > 1) {
            std::fprintf(stderr, "max_swap_n_ %u set. Searching multiswaps\n", max_swap_n_);
            run_multi(max_swap_n_);
        }
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
auto make_kmed_lsearcher(const Mat &mat, unsigned k, double eps=0.01, uint64_t seed=0,
                         const IndexContainer *wc=nullptr, double initdiv=0.) {
    return LocalKMedSearcher<Mat, IType>(mat, k, eps, seed, wc, initdiv);
}

} // graph
using graph::make_kmed_esearcher;
using graph::make_kmed_lsearcher;
using graph::LocalKMedSearcher;
using graph::ExhaustiveSearcher;


} // minocore

#endif /* FGC_LOCAL_SEARCH_H__ */
