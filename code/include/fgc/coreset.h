#pragma once
#include <vector>
#include <map>
#define ALIAS_THREADSAFE 1
#include "alias_sampler/alias_sampler.h"
#include "shared.h"
#include <zlib.h>
#ifdef _OPENMP
#  include <omp.h>
#endif



namespace coresets {

using namespace shared;

enum SensitivityMethod: int {
    BRAVERMAN_FELDMAN_LANG, // 2016, New Frameowkrs
    FELDMAN_LANGBERG,       // 2011, Unified Framework
    LUCIC_FAULKNER_KRAUSE_FELDMAN, // 2017, Training Gaussian Mixture Models at Scale
    // aliases
    BFL=BRAVERMAN_FELDMAN_LANG,
    FL=FELDMAN_LANGBERG,
    LFKF=LUCIC_FAULKNER_KRAUSE_FELDMAN
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
    std::unique_ptr<FT []>   weights_;

    size_t                        np_;
    uint64_t                    seed_ = 1337;
    bool ready() const {return sampler_.get();}

    CoresetSampler(CoresetSampler &&o)      = default;
    CoresetSampler(const CoresetSampler &o) = delete;
    CoresetSampler(): weights_(nullptr) {}

    void write(gzFile fp) const {
        uint64_t n = np_;
        gzwrite(fp, &n, sizeof(n));
        gzwrite(fp, &seed_, sizeof(seed_));
        gzwrite(fp, &probs_[0], sizeof(probs_[0]) * np_);
        uint32_t weights_present = weights_ ? 1337: 0;
        gzwrite(fp, &weights_present, sizeof(weights_present));
        if(weights_)
            gzwrite(fp, &weights_[0], sizeof(weights_[0]) * np_);
    }
    void write(std::FILE *fp) const {
        auto fd = ::fileno(fp);
        uint64_t n = np_;
        ::write(fd, &n, sizeof(n));
        ::write(fd, &seed_, sizeof(seed_));
        ::write(fd, &probs_[0], sizeof(probs_[0]) * np_);
        uint32_t weights_present = weights_ ? 1337: 0;
        ::write(fd, &weights_present, sizeof(weights_present));
        if(weights_)
            ::write(fd, &weights_[0], sizeof(weights_[0]) * np_);
    }
    void read(gzFile fp) {
        uint64_t n;
        gzread(fp, &n, sizeof(n));
        gzread(fp, &seed_, sizeof(seed_));
        probs_.reset(new FT[n]);
        gzread(fp, &probs_[0], sizeof(FT) * n);
        uint32_t weights_present;
        gzread(fp, &weights_present, sizeof(weights_present));
        if(weights_present) {
            assert(weights_present == 1337);
            weights_.reset(new FT[n]);
            gzread(fp, &weights_[0], sizeof(FT) * n);
        }
        sampler_.reset(new Sampler(probs_.get(), probs_.get() + n, seed_));
    }
    void read(std::FILE *fp) {
        uint64_t n;
        auto fd = ::fileno(fp);
        ::read(fd, &n, sizeof(n));
        ::read(fd, &seed_, sizeof(seed_));
        probs_.reset(new FT[n]);
        ::read(fd, &probs_[0], sizeof(FT) * n);
        uint32_t weights_present;
        ::read(fd, &weights_present, sizeof(weights_present));
        if(weights_present) {
            assert(weights_present == 1337);
            weights_.reset(new FT[n]);
            ::read(fd, &weights_[0], sizeof(FT) * n);
        }
        sampler_.reset(new Sampler(probs_.get(), probs_.get() + n, seed_));
    }

    void make_gmm_sampler(size_t np, size_t ncenters,
                      const FT *costs, const IT *assignments,
                      const FT *weights=nullptr,
                      uint64_t seed=137,
                      double alpha_est=0.)
    {
        // Note: this takes actual distances and then squares them.
        // ensure that the costs provided are L2Norm, not sqrL2Norm.
        // From Training Gaussian Mixture Models at Scale via Coresets
        // http://www.jmlr.org/papers/volume18/15-506/15-506.pdf
        // Note: this can be expanded to general probability measures.
        // I should to generalize this, such as for negative binomial and/or fancier functions
        np_ = np;
        if(weights) {
            weights_.reset(new FT[np]);
            std::memcpy(&weights_[0], weights, sizeof(FT) * np);
        } else weights_.release();
        std::vector<FT> weight_sums(ncenters), weighted_cost_sums(ncenters);
        std::vector<FT> sqcosts(ncenters);
        std::vector<IT> center_counts(ncenters);
        double total_cost = 0.;

        // TODO: vectorize
        // Sketch: check for SSE2/AVX2/AVX512
        //         if weights is null,
        //         set a vector via Space::set1()
        //
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
            auto cost = costs[i] * costs[i]; // d^2(x, A)
            auto wcost = w * cost;
            OMP_PRAGMA("omp atomic")
            weighted_cost_sums[asn] += wcost;
            OMP_PRAGMA("omp atomic")
            weight_sums[asn] += w; // If unweighted, weights are 1.
            OMP_PRAGMA("omp atomic")
            ++center_counts[asn];
            total_cost += wcost;
            sqcosts[i] = cost;
        }
        OMP_PRAGMA("omp parallel for")
        for(size_t i = 0; i < np; ++i) {
            this->probs_[i] = alpha_est * getweight(i) * (sqcosts[i] + weighted_cost_sums[assignments[i]] / weight_sums[assignments[i]])
                        + 2. * total_cost / weight_sums[assignments[i]];
        }
        auto si = 1. / std::accumulate(&this->probs_[0], this->probs_.get() + np, 0.);
        OMP_PRAGMA("omp parallel for")
        for(size_t i = 0; i < np; ++i)
            this->probs_[i] *= si;
        sampler_.reset(new Sampler(probs_.get(), probs_.get() + np, seed));
    }
    void make_sampler(size_t np, size_t ncenters,
                      const FT *costs, const IT *assignments,
                      const FT *weights=nullptr,
                      uint64_t seed=137,
                      SensitivityMethod sens=BRAVERMAN_FELDMAN_LANG,
                      double alpha_est=0.)
    {
        seed_ = seed;
        if(sens == LUCIC_FAULKNER_KRAUSE_FELDMAN) {
            make_gmm_sampler(np, ncenters, costs, assignments, weights, seed, alpha_est);
            return;
        }
        np_ = np;
        if(weights) {
            weights_.reset(new FT[np]);
            std::memcpy(&weights_[0], weights, sizeof(FT) * np);
        } else weights_.release();
        std::vector<FT> weight_sums(ncenters);
        std::vector<IT> center_counts(ncenters);
        double total_cost = 0.;

        // TODO: vectorize
        // Sketch: check for SSE2/AVX2/AVX512
        //         if weights is null,
        //         set a vector via Space::set1()
        //
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
            OMP_PRAGMA("omp atomic")
            ++center_counts[asn];
            total_cost += w * costs[i];
        }
        const bool is_feldman = (sens == FELDMAN_LANGBERG);
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
                probs_[i] = getweight(i) * (costs[i] * tcinv + weight_sums[assignments[i]]); // Am I propagating weights correctly?
            }
            sampler_.reset(new Sampler(probs_.get(), probs_.get() + np, seed));
        }
    }
    auto getweight(size_t ind) const {
        return weights_ ? weights_[ind]: static_cast<FT>(1.);
    }
    IndexCoreset<IT, FT> sample(const size_t n, uint64_t seed=0) {
        if(!sampler_.get()) throw 1;
        seed = seed ? seed: seed_;
        sampler_->seed(seed);
        IndexCoreset<IT, FT> ret(n);
#ifdef ALIAS_THREADSAFE
        auto ptr = &ret.indices_[0];
        size_t end = (n / 8) * 8;
        OMP_PRAGMA("omp parallel for")
        for(size_t i = 0; i < end; i += 8) {
            ptr[i + 0] = sampler_->sample();
            ptr[i + 1] = sampler_->sample();
            ptr[i + 2] = sampler_->sample();
            ptr[i + 3] = sampler_->sample();
            ptr[i + 4] = sampler_->sample();
            ptr[i + 5] = sampler_->sample();
            ptr[i + 6] = sampler_->sample();
            ptr[i + 7] = sampler_->sample();
        }
        while(end < n)
            ptr[end++] = sampler_->sample();
#else
        SK_UNROLL_8
        for(size_t i = 0; i < n; ++i) {
            ret.indices_[i] = sampler_->sample();
        }
#endif
#ifndef NDEBUG
        for(auto i = &ret.indices_[0]; i != &ret.indices_[n]; ++i)
            assert(size_t(i - &ret.indices_[0]) < np_);
#endif
        double nsamplinv = 1. / n;
        OMP_PRAGMA("omp parallel for")
        for(size_t i = 0; i < n; ++i)
            ret.weights_[i] = getweight(ret.indices_[i]) * nsamplinv / probs_[i];
        return ret;
    }
};


}//coresets


