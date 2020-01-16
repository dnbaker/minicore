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

template<typename Graph>
DiskMat<typename Graph::edge_property_type::value_type> graph2diskmat(const Graph &x, std::string path) {
    static_assert(std::is_arithmetic<typename Graph::edge_property_type::value_type>::value, "This should be floating point, or at least arithmetic");
    using FT = typename Graph::edge_property_type::value_type;
    auto nv = boost::num_vertices(x);
    DiskMat<FT> ret(nv, nv, path);
    //std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> vertices(boost::vertexs(x).first, boost::vertexs(x).second);
    typename boost::graph_traits<Graph>::vertex_iterator vertices = boost::vertices(x).first;
    const size_t e = boost::num_vertices(x);
    OMP_PFOR
    for(size_t i = 0; i < e; ++i) {
        auto mr = row(~ret, i);
        boost::dijkstra_shortest_paths(x, vertices[i], distance_map(&mr[0]));
    }
    assert((~ret).rows() == nv && (~ret).columns() == nv);
    return ret;
}

#if 0
template<typename Mat, typename Norm>
struct MatrixIndexNorm {
    const Mat mat_;
    const Norm norm_;
    template<typename Mat>
    MatrixMetric(const Mat &mat): mat_(mat) {}
    template<typename AT>
    auto operator()(size_t i, size_t j) const {
        return mat_(i, j);
    }
};
#endif

template<typename MatType, typename WFT=float, typename IType=std::uint32_t>
struct LocalKMedSearcher {
    using SolType = blaze::SmallArray<IType, 16>;
    using value_type = typename MatType::ElementType;


    const MatType &mat_;
    shared::flat_hash_set<IType> sol_;
    blz::DV<IType> assignments_;
    //std::vector<std::vector<IType>> center_indices_;
    //blz::DV<value_type> costs_;
    //SolType counts_;
    double current_cost_;
    const double eps_;
    IType k_;
    const size_t nr_, nc_;

    // Constructors

    LocalKMedSearcher(const LocalKMedSearcher &o) = default;
    LocalKMedSearcher(LocalKMedSearcher &&o) {
        auto ptr = reinterpret_cast<const uint8_t *>(this);
        std::memset(ptr, 0, sizeof(*this));
        std::swap_ranges(ptr, ptr + sizeof(*this), reinterpret_cast<const uint8_t *>(std::addressof(o)));
    }
    LocalKMedSearcher(const MatType &mat, unsigned k, double eps=0.01, uint64_t seed=0):
        mat_(mat), assignments_(mat.columns(), 0),
        // center_indices_(k),
        //costs_(mat.columns(), std::numeric_limits<value_type>::max()),
        //counts_(k),
        current_cost_(std::numeric_limits<value_type>::max()),
        eps_(eps),
        k_(k), nr_(mat.rows()), nc_(mat.columns())
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
            auto r = row(mat_, center, blaze::unchecked);
            OMP_PFOR
            for(size_t ci = 0; ci < nc_; ++ci) {
                assert(ci < r.size());
                if(const auto newcost = r[ci];
                   newcost < mat_(assignments_[ci], ci))
                {
                    //OMP_CRITICAL
                    {
                        assert(ci < assignments_.size());
                        assert(assignments_[ci] < nr_ || assignments_[ci] == IType(-1));
                        if(newcost < mat_(assignments_[ci], ci)) {
                            assignments_[ci] = center;
                        }
                    }
                }
            }
        }
#ifndef NDEBUG
#if 0
        for(size_t i = 0; i < nc_; ++i) {
            std::fprintf(stderr, "index %zu. cost: %f. assignment: %zu\n", i, costs_[i], size_t(assignments_[i]));
            assert(mat_(assignments_[i], i) == costs_[i]);
        }
#endif
        for(const auto asn: assignments_) {
            assert(asn < nr_);
        }
#endif
        assert(assignments_.size() == nc_);
        current_cost_ = cost_for_sol(sol_);
#if 0
        std::fill(counts_.begin(), counts_.end(), IType(0));
        //OMP_PFOR
        for(size_t ci = 0; ci < nc_; ++ci) {
            //OMP_ATOMIC
            auto asn = assignments_[ci];
            ++counts_[asn];
            blz::push_back(center_indices_.at(asn), ci); // replace at with operator[] if this works as expected
        }
        assert(std::accumulate(counts_.begin(), counts_.end(), 0u) == nc_ || !std::fprintf(stderr, "Doesn't add up: %u\n", std::accumulate(counts_.begin(), counts_.end(), 0u)));
        std::fprintf(stderr, "current cost: %f\n", current_cost_);
        assert(counts_.size() == sol_.size());
        for(size_t i = 0; i < counts_.size(); ++i) {
            std::fprintf(stderr, "count %u is %zu\n", unsigned(sol_[i]), size_t(counts_[i]));
        }
        std::fputc('\n', stderr);
#endif
    }

    double evaluate_swap(IType newcenter, IType oldcenter) const {
        assert(newcenter < mat_.rows());
        assert(oldcenter < mat_.rows());
        auto newr = row(mat_, newcenter, blaze::unchecked);
        double potential_gain = 0.;
        OMP_PRAGMA("omp parallel for reduction(+:potential_gain)")
        for(size_t i = 0; i < nc_; ++i) {
            //if(assignments_[i] != oldcenter) continue; // Only points already assigned to
            //assert(costs_[i] == mat_(assignments_[i], i));
            auto asn = assignments_[i];
            if(asn == oldcenter || newr[i] < mat_(asn, i)) {
                potential_gain += mat_(asn, i) - newr[i];
            }
        }
        // TODO: make sure this is right.
        return potential_gain;
    }

#if 0
    double evaluate_swap_lazy(IType newcenter, IType oldcenterindex, const std::vector<IType> &c) const {
        // oldcenter is the index into the sol_ array
        // This function ignores the fact that the new center could outperform other centers
        //std::fprintf(stderr, "I think this might be wrong...\n");
        //throw std::runtime_error("Don't permit calling this currently.\n");
        assert(newcenter < mat_.rows());
        const auto oldcenter = sol_[oldcenterindex];
        auto newr = row(mat_, newcenter, blaze::unchecked);
        auto oldr = row(mat_, oldcenter, blaze::unchecked);
        //const size_t nc = mat_.columns();
        double potential_gain = 0.;
        const auto csz = c.size();
        OMP_PRAGMA("omp parallel for reduction(+:potential_gain)")
        for(size_t i = 0; i < csz; ++i) {
            const auto ind = c[i];
            assert(assignments_[ind] == oldcenterindex);
            assert(sol_[oldcenterindex] == oldcenter);
            potential_gain += (oldr[ind] - newr[ind]);
        }
        return potential_gain;
    }
#endif

    // Getters
    auto k() const {
        return k_;
    }

#if VERBOSE_AF
    void recalculate(IType oldcenter, IType newcenter) {
        std::fprintf(stderr, "swapping out oldcenter %u for %u\n", oldcenter, newcenter);
        assert(std::find(sol_.begin(), sol_.end(), oldcenter) == sol_.end());
#else
    void recalculate() {
#endif
        // This function should:
        // 1. Update costs_
        // 2. Update current_cost_
        // 3. Update center_indices_
        //auto oldcenter = sol_[oldcenterindex];
        //pdqsort(sol_.begin(), sol_.end());
        assignments_ = *sol_.begin();

        OMP_PFOR
        for(size_t i = 0; i < nc_; ++i) {
            for(auto it = sol_.begin(); ++it != sol_.end();) {
                const auto center = *it;
                if(const auto newcost(mat_(center, i));
                   newcost < mat_(assignments_[i], i))
                {
                    OMP_CRITICAL
                    {
                    if(newcost < mat_(assignments_[i], i))
                        assignments_[i] = center;
                    }
                }
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
#if 1
        std::fprintf(stderr, "newcost: %f. old cost: %f\n", newcost, current_cost_);
#endif
        assert(newcost <= current_cost_);
        current_cost_ = newcost;
    }

    void run() {
        //auto it = std::max_element(cbeg, cd, [](const auto &x, const auto &y) {return x.size() < y.size();});

        double diffthresh = current_cost_ / k_ * eps_;
        size_t total = 0;
        {
            next:
            for(const auto oldcenter: sol_) {
                for(size_t pi = 0; pi < nr_; ++pi) {
                    if(sol_.find(pi) != sol_.end()) continue;
                    const auto val = evaluate_swap(pi, oldcenter);
                    if(val > diffthresh) {
                        std::fprintf(stderr, "Swapping %zu for %u\n", pi, oldcenter);
                        sol_.erase(oldcenter);
                        sol_.insert(pi);
#if VERBOSE_AF
                        recalculate(oldcenter, pi);
#else
                        recalculate();
#endif
                        diffthresh = current_cost_ / k_ * eps_;
                        ++total;
                        goto next; // Meaning we've swapped this guy out and will pick another one.
                    }
                }
            }
        }
#if VERBOSE_AF
        const auto nc = cost_for_sol(sol_);
        assert(std::abs(current_cost_ - nc) < 1e-10);
        std::fprintf(stderr, "Cost by manual: %f. cost using exhaustive; %f\n", current_cost_, nc);
        current_cost_ = nc;
#endif
        std::fprintf(stderr, "Finished in %zu swaps by exhausting all potential improvements. Final cost: %f\n",
                     total, current_cost_);
    }

    // Steps:
    // 1. Use k-center approx for seeds
    // 2. Loop over finding candidate replacements and performing swaps.
};

template<typename Mat, typename FT=float, typename IType=std::uint32_t>
auto make_kmed_lsearcher(const Mat &mat, unsigned k, double eps=0.01, uint64_t seed=0) {
    return LocalKMedSearcher<Mat, FT, IType>(mat, k, eps, seed);
}

template<typename MT, blaze::AlignmentFlag AF, bool SO, bool DF, typename IType=std::uint32_t>
auto make_kmed_lsearcher(const blaze::Submatrix<MT, AF, SO, DF> &mat, unsigned k, double eps=0.01, uint64_t seed=0) {
    using FT = typename MT::ElementType;
    blaze::CustomMatrix<FT, AF, blaze::IsPadded<MT>::value, SO> custom(const_cast<FT *>(mat.data()), mat.rows(), mat.columns(), mat.spacing());
    return LocalKMedSearcher<decltype(custom), FT, IType>(custom, k, eps, seed);
}

} // fgc

#endif /* FGC_LOCAL_SEARCH_H__ */
