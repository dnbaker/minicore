#pragma once
#include <vector>
#include <map>
#include "aesctr/wy.h"
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
    std::vector<IT> indices_;
    std::vector<FT> weights_;
    size_t size() const {return indices_.size();}
    Coreset(Coreset &&o) = default;
    Coreset(const Coreset &o) = default;
    void compact(bool shrink_to_fit=false) {
        // TODO: replace with hash map and compact
        std::map<std::pair<IT, FT>, uint32_t> m;
        for(IT i = 0; i < indices_.size(); ++i) {
            ++m[std::make_pair(indices_[i], weights_[i])];
            //++m[std::make_pair(p.first, p.second)];
        }
        if(m.size() == indices_.size()) return;
        auto it = &indices_[0];
        auto wit = &weights_[0];
        for(const auto &pair: m) {
            *it++ = pair.first.first;
            *wit++ = pair.second * pair.first.second; // Add the weights together
        }
        size_t newsz = it - &indices_[0];
        indices_.resize(newsz);
        weights_.resize(newsz);
        if(shrink_to_fit) indices_.shrink_to_fit(), weights_.shrink_to_fit();
    }
    std::vector<std::pair<IT, FT>> to_pairs() const {
        std::vector<std::pair<IT, FT>> ret;
        ret.reserve(size());
        for(IT i = 0; i < size(); ++i)
            ret.push_back(std::make_pair(indices_[i], weights_[i]));
        return ret;
    }
    void show() {
        for(size_t i = 0; i < indices_.size(); ++i) {
            std::fprintf(stderr, "%zu: [%u/%g]\n", i, indices_[i], weights_[i]);
        }
    }
    Coreset(size_t n): indices_(n), weights_(n) {}
#if 0
    struct iterator {
        // TODO: operator++, operator++(int), operator--
        Coreset &ref_;
        size_t index_;
        iterator(Coreset &ref, size_t index): ref_(ref), index_(index) {}
        using deref_type = std::pair<std::reference_wrapper<IT>, std::reference_wrapper<FT>>;
        deref_type operator*() {
            return deref_type(std::ref(ref_.indices_[index_]), std::ref(ref_.weights_[index_]));
        }
        iterator(iterator &o): ref_(o.ref_), index_(o.index_) {}
        iterator &operator++() {
            ++index_;
            return *this;
        }
        iterator operator++(int) {
            iterator ret(*this);
            ++index_;
            return ret;
        }
        bool operator==(iterator o) const {
            return index_ == o.index_;
        }
        bool operator!=(iterator o) const {
            return index_ != o.index_;
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
        const_iterator(const_iterator &o): ref_(o.ref_), index_(o.index_) {}
        const_iterator &operator++() {
            ++index_;
            return *this;
        }
        const_iterator operator++(int) {
            const_iterator ret(*this);
            ++index_;
        }
        bool operator==(const_iterator o) const {
            return index_ == o.index_;
        }
        bool operator!=(const_iterator o) const {
            return index_ != o.index_;
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
        return iterator(*this, size());
    }
    auto end() const {
        return const_iterator(*this, size());
    }
#endif
};

template<typename FT=float, typename IT=std::uint32_t>
struct CoresetSampler {
    using Sampler = alias::AliasSampler<FT, wy::WyRand<IT, 2>, IT>;
    std::unique_ptr<Sampler> sampler_;
    std::unique_ptr<FT []>     probs_;
    const FT                *weights_;
    size_t                        np_;
    bool ready() const {return sampler_.get();}

    CoresetSampler(CoresetSampler &&o)      = default;
    CoresetSampler(const CoresetSampler &o) = delete;
    CoresetSampler(): weights_(nullptr) {}

    void make_sampler(size_t np, size_t ncenters,
                      const FT *costs, const IT *assignments,
                      const FT *weights=nullptr,
                      uint64_t seed=137)
    {
        np_ = np;
        weights_ = weights;
        std::vector<FT> weight_sums(ncenters);
        std::vector<IT> center_counts(ncenters);
        FT total_cost = 0.;

        for(size_t i = 0; i < np; ++i) {
            // TODO: vectorize?
            // weight sums per assignment couldn't be vectorized,
            // total costs could be
            // Probably a 4-16x speedup on 1/3 of the cost
            // So maybe like a ~30% speedup?
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
        sampler_->operator()(&ret.indices_[0], &ret.indices_[n], n ^ seed);
        for(auto i = &ret.indices_[0]; i != &ret.indices_[n]; ++i)
            assert(size_t(i - &ret.indices_[0]) < np_);
        double nsamplinv = 1. / n;
        for(size_t i = 0; i < n; ++i)
            ret.weights_[i] = getweight(ret.indices_[i]) * nsamplinv / probs_[i];
        return ret;
    }
};



}

}//coresets


