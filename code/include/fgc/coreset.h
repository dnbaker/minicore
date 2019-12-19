#pragma once
#include <vector>
#include <map>
#include "alias_sampler/alias_sampler.h"
#include "shared.h"
#ifdef _OPENMP
#  include <omp.h>
#endif



namespace coresets {

using namespace shared;

enum SensitivityMethod: int {
    BRAVERMAN_FELDMAN_LANG,
    FELDMAN_LANGBERG,
    // aliases
    BFL=BRAVERMAN_FELDMAN_LANG,
    FL=FELDMAN_LANGBERG
};

template<typename IT, typename FT>
struct IndexCoreset {
    static_assert(std::is_integral<IT>::value, "IT must be integral");
    static_assert(std::is_floating_point<FT>::value, "FT must be floating-point");
    /*
     * consists of only indices and weights
     */
    std::vector<IT> indices_;
    std::vector<FT> weights_;
    size_t size() const {return indices_.size();}
    IndexCoreset(IndexCoreset &&o) = default;
    IndexCoreset(const IndexCoreset &o) = default;
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
        std::vector<std::pair<IT, FT>> ret(size());
        OMP_PRAGMA("omp parallel for")
        for(IT i = 0; i < size(); ++i)
            ret[i].first = indices_[i], ret[i].second = weights_[i];
        return ret;
    }
    void show() {
        for(size_t i = 0; i < indices_.size(); ++i) {
            std::fprintf(stderr, "%zu: [%u/%g]\n", i, indices_[i], weights_[i]);
        }
    }
    IndexCoreset(size_t n): indices_(n), weights_(n) {}
};

#if 0
#ifdef __AVX512F__
#define USE_VECTORS
#define VECTOR_WIDTH 64u
template<typename FT>
struct VT;
template<> VT<float>{using type = __m512;}
template<> VT<double>{using type = __m512d;}
#elif defined(__AVX2__)
template<typename FT>
struct VT;
template<> VT<float>{using type = __m512;}
template<> VT<double>{using type = __m512d;}
#define VECTOR_WIDTH 32u
#elif defined(__SSE2__)
#define VECTOR_WIDTH 16u
#endif
#endif

template<typename FT=float, typename IT=std::uint32_t>
struct CoresetSampler {
    using Sampler = alias::AliasSampler<FT, wy::WyRand<IT, 2>, IT>;
    std::unique_ptr<Sampler> sampler_;
    std::unique_ptr<FT []>     probs_;

    SensitivityMethod sm_ = BRAVERMAN_FELDMAN_LANG;
    const FT                *weights_;
    size_t                        np_;
    bool ready() const {return sampler_.get();}

    CoresetSampler(CoresetSampler &&o)      = default;
    CoresetSampler(const CoresetSampler &o) = delete;
    CoresetSampler(): weights_(nullptr) {}

    void make_sampler(size_t np, size_t ncenters,
                      const FT *costs, const IT *assignments,
                      const FT *weights=nullptr,
                      uint64_t seed=137,
                      SensitivityMethod sens=BRAVERMAN_FELDMAN_LANG)
    {
        np_ = np;
        weights_ = weights;
        std::vector<FT> weight_sums(ncenters);
        std::vector<IT> center_counts(ncenters);
        double total_cost = 0.;
        sm_ = sens;

        // TODO: vectorize
        // Sketch: check for SSE2/AVX2/AVX512
        //         if weights is null,
        //         set a vector via Space::set1()
        //
#if 0
// #ifdef VECTOR_WIDTH
        /* under construction */
        static constexpr unsigned nelem_per_vector =
            VECTOR_WIDTH == unsigned(VECTOR_WIDTH / sizeof(FT));
        using VTP = typename VT<FT>::type;
        VTP
        const VTP *vw = reinterpret_cast<
        OMP_PRAGMA("omp parallel for reduction(+:total_cost)")
        for(size_t i = 0; i < np / VECTOR_WIDTH; ++i) {
            VT v(/*load vector */)
            /* get assignments */
            /* add weight sum */
        }
        for(size_t i = (np / VECTOR_WIDTH) * VECTOR_WIDTH;i < np; ++i) {
        }
#else
        OMP_PRAGMA("omp parallel for reduction(+:total_cost)")
        for(size_t i = 0; i < np; ++i) {
            // TODO: vectorize?
            // weight sums per assignment couldn't be vectorized,
            // total costs could be
            // Probably a 4-16x speedup on 1/3 of the cost
            // So maybe like a ~30% speedup?
            auto asn = assignments[i];
            assert(asn < ncenters);
            const auto w = getweight(i);

            OMP_PRAGMA("omp atomic")
            weight_sums[asn] += w; // If unweighted, weights are 1.
            ++center_counts[asn];
            total_cost += w * costs[i];
        }
#endif
        const bool is_feldman = (sm_ == FELDMAN_LANGBERG);
        if(is_feldman)
            total_cost *= 2.; // For division
        const auto tcinv = 1. / total_cost;
        probs_.reset(new FT[np]);
        // TODO: Consider log space?
        if(is_feldman) {
            // Ignores number of items assigned to each cluster
            // std::fprintf(stderr, "note: FL method has worse guarantees than BFL\n");
            sampler_.reset(new Sampler(probs_.get(), probs_.get() + np, seed));
            OMP_PRAGMA("omp parallel for")
            for(size_t i = 0; i < np; ++i) {
                probs_[i] = getweight(i) * (costs[i]) * tcinv;
            }
        } else {
            for(auto i = 0u; i < ncenters; ++i)
                weight_sums[i] = 1./(2. * center_counts[i] * weight_sums[i]);
            OMP_PRAGMA("omp parallel for")
            for(size_t i = 0; i < np; ++i) {
                probs_[i] = getweight(i) * (costs[i] * tcinv + weight_sums[assignments[i]]);
            }
            sampler_.reset(new Sampler(probs_.get(), probs_.get() + np, seed));
        }
    }
    auto getweight(size_t ind) const {
        return weights_ ? weights_[ind]: static_cast<FT>(1.);
    }
    IndexCoreset<IT, FT> sample(size_t n, uint64_t seed=0) {
        IndexCoreset<IT, FT> ret(n);
        sampler_->operator()(&ret.indices_[0], &ret.indices_[n], n ^ seed);
        for(auto i = &ret.indices_[0]; i != &ret.indices_[n]; ++i)
            assert(size_t(i - &ret.indices_[0]) < np_);
        double nsamplinv = 1. / n;
        OMP_PRAGMA("omp parallel for")
        for(size_t i = 0; i < n; ++i)
            ret.weights_[i] = getweight(ret.indices_[i]) * nsamplinv / probs_[i];
        return ret;
    }
};


}//coresets


