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
    using DType = typename MatType::ElementType;
    using SolType = blaze::SmallArray<IType, 16>;


    const MatType &mat_;
    SolType sol_;
    std::vector<IType> assignments_;
    std::vector<std::vector<IType>> center_indices_;
    blz::DV<DType> costs_;
    SolType counts_;
    double current_cost_;
    const double eps_;
    IType k_;

    // Constructors

    LocalKMedSearcher(const LocalKMedSearcher &o) = default;
    LocalKMedSearcher(LocalKMedSearcher &&o) {
        auto ptr = reinterpret_cast<const uint8_t *>(this);
        std::memset(ptr, 0, sizeof(*this));
        std::swap_ranges(ptr, ptr + sizeof(*this), reinterpret_cast<const uint8_t *>(std::addressof(o)));
    }
    LocalKMedSearcher(const MatType &mat, unsigned k, double eps=0.01):
        mat_(mat), assignments_(mat.columns(), IType(-1)), center_indices_(k),
        costs_(mat.columns(), std::numeric_limits<DType>::max()),
        counts_(k),
        current_cost_(std::numeric_limits<DType>::max()),
        eps_(eps),
        k_(k)
    {
        wy::WyRand<IType, 2> rng(k / eps * mat.rows() + mat.columns());
        auto rowits = rowiterator(mat);
        auto approx = coresets::kcenter_greedy_2approx(rowits.begin(), rowits.end(), rng, k, MatrixLookup());
        std::fprintf(stderr, "Approx is complete with size %zu (expected k: %u)\n", approx.size(), k);
        sol_.resize(k);
        std::copy(approx.begin(), approx.end(), sol_.begin());
        pdqsort(sol_.data(), sol_.data() + sol_.size()); // Just for access pattern
        assign();
    }

    // Setup/Utilities

    void assign() {
        const size_t nc = mat_.columns(), nr = mat_.rows(), ncand = sol_.size();
        assert(assignments_.size() == nc);
        OMP_PFOR
        for(size_t ri = 0; ri < ncand; ++ri) {
            assert(sol_[ri] < mat_.rows());
            auto r = row(mat_, sol_[ri], blaze::unchecked);
            for(size_t ci = 0; ci < nc; ++ci) {
                assert(ci < r.size());
                if(const auto newcost = r[ci];
                   newcost < costs_[ci])
                {
                    OMP_CRITICAL
                    {
                        if(newcost < costs_[ci]) {
                            costs_[ci] = newcost;
                            assignments_[ci] = ri;
                        }
                    }
                }
            }
        }
#ifndef NDEBUG
        for(size_t i = 0; i < nc; ++i) {
            std::fprintf(stderr, "index %zu. cost: %f. assignment: %zu\n", i, costs_[i], size_t(assignments_[i]));
        }
        for(const auto asn: assignments_)
            assert(asn < nr);
#endif
        assert(assignments_.size() == nc);
        current_cost_ = blaze::sum(costs_);
        std::fill(counts_.begin(), counts_.end(), IType(0));
        //OMP_PFOR
        for(size_t ci = 0; ci < nc; ++ci) {
            //OMP_ATOMIC
            auto asn = assignments_[ci];
            ++counts_[asn];
            blz::push_back(center_indices_.at(asn), ci); // replace at with operator[] if this works as expected
        }
        assert(std::accumulate(counts_.begin(), counts_.end(), 0u) == nc || !std::fprintf(stderr, "Doesn't add up: %u\n", std::accumulate(counts_.begin(), counts_.end(), 0u)));
        std::fprintf(stderr, "current cost: %f\n", current_cost_);
        assert(counts_.size() == sol_.size());
#if VERBOSE_AF
        for(size_t i = 0; i < counts_.size(); ++i) {
            std::fprintf(stderr, "count %u is %zu\n", unsigned(sol_[i]), size_t(counts_[i]));
        }
        std::fputc('\n', stderr);
#endif
    }

    double evaluate_swap(IType newcenter, IType oldcenterindex) {
        // oldcenter is the index into the sol_ array
        assert(oldcenterindex < sol_.size());
        assert(newcenter < mat_.rows());
        const auto oldcenter = sol_[oldcenterindex];
        auto newr = row(mat_, newcenter, blaze::unchecked);
        auto oldr = row(mat_, oldcenter, blaze::unchecked);
        const size_t nc = mat_.columns();
        double potential_gain = 0.;
        OMP_PRAGMA("omp parallel for reduction(+:potential_gain)")
        for(size_t i = 0; i < nc; ++i) {
            if(assignments_[i] != oldcenter) continue; // Only points already assigned to
            if(assignments_[i] == oldcenter) {
                potential_gain += (oldr[i] - newr[i]);
            } else if(newr[i] < costs_[i]) {
                potential_gain += costs_[i] - newr[i];
            }
        }
        return potential_gain;
    }


    double evaluate_swap_lazy(IType newcenter, IType oldcenterindex, const std::vector<IType> &c) const {
        // oldcenter is the index into the sol_ array
        // This function ignores the fact that the new center could outperform other centers
        //std::fprintf(stderr, "I think this might be wrong...\n");
        //throw std::runtime_error("Don't permit calling this currently.\n");
        assert(newcenter < mat_.rows());
        const auto oldcenter = sol_[oldcenterindex];
        auto newr = row(mat_, newcenter, blaze::unchecked);
        auto oldr = row(mat_, oldcenter, blaze::unchecked);
        const size_t nc = mat_.columns();
        double potential_gain = 0.;
        const auto csz = c.size();
        OMP_PRAGMA("omp parallel for reduction(+:potential_gain)")
        for(size_t i = 0; i < csz; ++i) {
            const auto ind = c[i];
            assert(assignments_[ind] == oldcenterindex);
            potential_gain += (oldr[ind] - newr[ind]);
        }
        return potential_gain;
    }

    // Getters
    auto k() const {
        return k_;
    }

    void perform_swap(IType oldcenterindex, IType newcenter) {
        std::fprintf(stderr, "swapping out oldcenter %u at index %u for %u\n", sol_[oldcenterindex], oldcenterindex, newcenter);
        // This function should:
        // 1. Update costs_
        // 2. Update current_cost_
        // 3. Update center_indices_
        const size_t nc = mat_.columns();
        //auto oldcenter = sol_[oldcenterindex];
        sol_[oldcenterindex] = newcenter;

        auto newr = row(mat_, newcenter);

        // Calculate 
        for(size_t i = 0; i < nc; ++i) {
            if(newr[i] < costs_[i]) {
#if 1
                if(assignments_[i] != oldcenterindex) {
                    assert(std::find(center_indices_[assignments_[i]].begin(), center_indices_[assignments_[i]].end(), i) != center_indices_[assignments_[i]].end());
                    assignments_[i] = oldcenterindex;
                }
#endif
                assignments_[i] = oldcenterindex;
                costs_[i] = newr[i];
            } else if(assignments_[i] == oldcenterindex) {
                double mincost = mat_(sol_[0], i);
                IType newasn = 0;
                for(IType j = 1, e = sol_.size(); j < e; ++j) {
                    if(auto mc = mat_(sol_[j], i); mc < mincost)
                        newasn = j, mincost = mc;
                }
                assignments_[i] = newasn;
            }
        }
        

#if 1
        for(auto &i: center_indices_) i.clear();
        for(size_t i = 0; i < nc; ++i)
            center_indices_[assignments_[i]].push_back(i);
#endif
        double newcost = 0.;
        OMP_PRAGMA("omp parallel for reduction(+:newcost)")
        for(size_t i = 0; i < nc; ++i)
            newcost += costs_[i];
        std::fprintf(stderr, "newcost: %f. old cost: %f\n", newcost, current_cost_);
        assert(newcost <= current_cost_);
        current_cost_ = newcost;
    }

    void run(size_t max_iter=30) {
        const size_t nr = mat_.rows();
        //auto it = std::max_element(cbeg, cd, [](const auto &x, const auto &y) {return x.size() < y.size();});
        blaze::SmallArray<IType, 16> sv(k_);
        //std::copy(sol_.begin(), sol_.end(), sv.begin());
        std::iota(sv.data(), sv.data() + k_, IType(0));
        auto cicmp = [&](auto x, auto y) {
             return center_indices_[x].size() < center_indices_[y].size();
        };
        pdqsort(sv.begin(), sv.end(), cicmp);
        ska::flat_hash_set<IType> current_centers(sol_.begin(), sol_.end());
        const bool linear_check = k_ < 80; // if k_ < 80, check linearly, otherwise use the hash set.
        double threshold = current_cost_ * (1. - eps_ / k_);
        double diffthresh = current_cost_ / k_ * eps_;
        bool exhausted_lazy, use_full_cmp = false;
        for(size_t iternum = 0; iternum < max_iter; ++iternum) {
            std::fprintf(stderr, "iternum: %zu\n", iternum);
            exhausted_lazy = true;
            for(const auto oldcenterindex: sv) {
                //std::fprintf(stderr, "oci: %u. ci size: %zu. sol size: %zu\n", oldcenterindex, center_indices_.size(), sol_.size());
                assert(oldcenterindex < sol_.size());
                const auto oldcenter = sol_[oldcenterindex];
                const auto &oldcenterindices = center_indices_[oldcenterindex];
                for(size_t pi = 0; pi < nr; ++pi) {
                    if(linear_check ? std::find(sol_.begin(), sol_.end(), pi) != sol_.end()
                                    : current_centers.find(pi) != current_centers.end())
                        continue;
                    auto val = use_full_cmp ? evaluate_swap(pi, oldcenterindex)
                                            : evaluate_swap_lazy(pi, oldcenterindex, oldcenterindices);
                    if(val > diffthresh) {
                        perform_swap(oldcenterindex, pi);
                        current_centers.erase(oldcenter);
                        current_centers.insert(pi);
                        sol_[oldcenterindex] = pi;
                        threshold = current_cost_ * (1. - eps_ / k_);
                        exhausted_lazy = false;
                        break; // Meaning we've swapped this guy out and will pick another one.
                    }
                }
            }
            if(exhausted_lazy) {
                if(use_full_cmp) break;
                use_full_cmp = true;
            }
            pdqsort(sv.begin(), sv.end(), cicmp);
        }
    }

    // Steps:
    // 1. Use k-center approx for seeds
    // 2. Loop over finding candidate replacements and performing swaps.
};

template<typename Mat, typename FT=float, typename IType=std::uint32_t>
auto make_kmed_lsearcher(const Mat &mat, unsigned k, double eps=0.01) {
    return LocalKMedSearcher<Mat, FT, IType>(mat, k, eps);
}

} // fgc

#endif /* FGC_LOCAL_SEARCH_H__ */
