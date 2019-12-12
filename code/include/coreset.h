#pragma once
#include "aesctr/wy.h"
#include <vector>
#include <map>
#include "robin-hood-hashing/src/include/robin_hood.h"
#include "alias_sampler/alias_sampler.h"

                                                                                                    

namespace coresets {
template <typename Key, typename T, typename Hash = robin_hood::hash<Key>,                             
          typename KeyEqual = std::equal_to<Key>, size_t MaxLoadFactor100 = 80>                        
using hash_map = robin_hood::unordered_flat_map<Key, T, Hash, KeyEqual, MaxLoadFactor100>;        
inline namespace sampling {

template<typename IT, typename FT>
struct Coreset {
    static_assert(std::is_integral<IT>::value, "IT must be integral");
    static_assert(std::is_floating_point<FT>::value, "FT must be floating-point");
    /*
     * consists of only indices and weights
     */
    std::unique_ptr<IT[]> indices_;
    std::unique_ptr<FT[]> weights_;
    size_t                      n_;
    size_t size() const {return n_;}
    Coreset(Coreset &&o) = default;
    Coreset(const Coreset &o): n_(o.n_) {
        indices_.reset(std::make_unique<IT[]>(n_));
        weights_.reset(std::make_unique<FT[]>(n_));
        std::memcpy(&indices_[0], &o.indices_[0], sizeof(IT) * n_);
        std::memcpy(&weights_[0], &o.weights_[0], sizeof(FT) * n_);
    }
    void compact(bool shrink_to_fit=false) {
        std::map<std::pair<IT, FT>, uint32_t> m;
        for(IT i = 0; i < n_; ++i) {
            ++m[std::make_pair(indices_[i], weights_[i])];
            //++m[std::make_pair(p.first, p.second)];
        }
        if(m.size() == n_) return;
        auto it = &indices_[0];
        auto wit = &weights_[0];
        for(const auto &pair: m) {
            *it++ = pair.first.first;
            *wit++ = pair.second * pair.first.second; // Add the weights together
        }
        n_ = m.size();
        if(shrink_to_fit) std::fprintf(stderr, "Note: not implemented. This shouldn't matter.\n");
    }
    Coreset(size_t n): indices_(std::make_unique<IT[]>(n)), weights_(std::make_unique<FT[]>(n)), n_(n) {}
    struct iterator {
        // TODO: operator++, operator++(int), operator--
        Coreset &ref_;
        size_t index_;
        iterator(Coreset &ref, size_t index): ref_(ref), index_(index) {}
        using deref_type = std::pair<std::reference_wrapper<IT>, std::reference_wrapper<FT>>;
        deref_type operator*() {
            return deref_type(std::ref(ref_.indices_[index_]), std::ref(ref_.weights_[index_]));
        }
        bool operator==(iterator o) const {
            return index_ == o.index_;
        }
        bool operator<(iterator o) const {
            return index_ < o.index_;
        }
    };
    struct const_iterator {
        const Coreset &ref_;
        size_t index_;
        const_iterator(const Coreset &ref, size_t index): ref_(ref), index_(index) {}
        using deref_type = std::pair<std::reference_wrapper<const IT>, std::reference_wrapper<const FT>>;
        deref_type operator*() {
            return deref_type(std::cref(ref_.indices_[index_]), std::cref(ref_.weights_[index_]));
        }
        bool operator==(const_iterator o) const {
            return index_ == o.index_;
        }
        bool operator<(const_iterator o) const {
            return index_ < o.index_;
        }
    };
    auto begin() {
        return iterator(*this, 0);
    }
    auto begin() const {
        return const_iterator(*this, 0);
    }
    auto end() {
        return iterator(*this, n_);
    }
    auto end() const {
        return const_iterator(*this, n_);
    }
};

template<typename FT=float, typename IT=std::uint32_t>
struct CoresetSampler {
    using Sampler = alias::AliasSampler<FT, wy::WyRand<IT, 2>, IT>;
    std::unique_ptr<Sampler> sampler_;
    std::unique_ptr<FT []>     probs_;
    const FT                *weights_;
    bool ready() const {return sampler_.get();}

    CoresetSampler(CoresetSampler &&o) = default;
    CoresetSampler(const CoresetSampler &o) = delete;
    CoresetSampler(): weights_(nullptr) {}
    
    void make_sampler(size_t np, size_t ncenters,
                      const FT *costs, const IT *assignments,
                      const FT *weights=nullptr,
                      uint64_t seed=137)
    {
        weights_ = weights;
        std::vector<FT> weight_sums(ncenters);
        std::vector<IT> center_counts(ncenters);
        FT total_cost = 0.;

        for(size_t i = 0; i < np; ++i) {
            auto asn = assignments[i];
            assert(asn < ncenters);
            const auto w = getweight(i);
            weight_sums[asn] += w; // If unweighted, weights are 1.
            total_cost += w * costs[i];
            ++center_counts[asn];
        }
        total_cost *= 2.; // For division
        auto tcinv = 1. / total_cost;
        probs_.reset(new FT[np]);
        for(auto i = 0u; i < ncenters; ++i)
            weight_sums[i] = 1./(2. * center_counts[i] * weight_sums[i]);
        for(size_t i = 0; i < np; ++i) {
            const auto w = getweight(i);
            probs_[i] = w * (costs[i] * tcinv + weight_sums[assignments[i]]);
        }
        sampler_.reset(new Sampler(probs_.get(), probs_.get() + np, seed));
    }
    auto getweight(size_t ind) const {
        return weights_ ? weights_[ind]: static_cast<FT>(1.);
    }
    Coreset<IT, FT> sample(size_t n, uint64_t seed=0) {
        Coreset<IT, FT> ret(n);
        sampler_->operator()(ret.indices_.get(), ret.indices_.get() + n, n ^ seed);
        double nsamplinv = 1. / n;
        for(size_t i = 0; i < n; ++i)
            ret.weights_[i] = getweight(ret.indices_[i]) * nsamplinv / probs_[i];
        return ret;
    }
};




}

}//coresets


