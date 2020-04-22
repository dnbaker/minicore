#pragma once
#ifndef FGC_CORESETS_H__
#define FGC_CORESETS_H__
#include <vector>
#include <map>
#include "alias_sampler/alias_sampler.h"
#include "minocore/util/blaze_adaptor.h"
#include "minocore/util/shared.h"
#include <zlib.h>
#ifdef _OPENMP
#  include <omp.h>
#endif
//#include "minocore/dist/distance.h"


namespace minocore {
namespace coresets {
#ifndef SMALLARRAY_DEFAULT_N
#define SMALLARRAY_DEFAULT_N 16
#endif

template<typename IT, typename Alloc=std::allocator<IT>, size_t N=SMALLARRAY_DEFAULT_N>
using IVec = blaze::SmallArray<IT, N, Alloc>;
template<typename FT=float, typename Alloc=std::allocator<FT>, size_t N=SMALLARRAY_DEFAULT_N>
using WVec = blaze::SmallArray<FT, N, Alloc>;

using namespace shared;

enum SensitivityMethod: int {
    BRAVERMAN_FELDMAN_LANG, // 2016, New Frameworks
    FELDMAN_LANGBERG,       // 2011, Unified Framework
    LUCIC_FAULKNER_KRAUSE_FELDMAN, // 2017, Training Gaussian Mixture Models at Scale
    VARADARAJAN_XIAO,              // 2012, On the Sensitivity of Shape-Fitting Problems
    LUCIC_BACHEM_KRAUSE,           // 2016, Strong Coresets for Hard and Soft Bregman Clustering with Applications to Exponential Family Mixtures
    // aliases
    BFL=BRAVERMAN_FELDMAN_LANG,
    FL=FELDMAN_LANGBERG,
    LFKF=LUCIC_FAULKNER_KRAUSE_FELDMAN,
    VX = VARADARAJAN_XIAO,
    BOUNDED_TREE_WIDTH = VX,
    BTW=BOUNDED_TREE_WIDTH,
    LBK=LUCIC_BACHEM_KRAUSE
};

static const char *sm2str(SensitivityMethod sm) {
    switch(sm) {
        case BFL:  return "BFL";
        case VX:   return "VX";
        case LFKF: return "LFKF";
        case LBK:  return "LBK";
        case FL:   return "FL";
    }
    return "UNKNOWN";
}

template<typename IT, typename FT>
struct IndexCoreset {
    static_assert(std::is_integral<IT>::value, "IT must be integral");
    static_assert(std::is_floating_point<FT>::value, "FT must be floating-point");
    /*
     * consists of only indices and weights
     */
    blaze::DynamicVector<IT> indices_;
    blaze::DynamicVector<FT> weights_;
    size_t size() const {return indices_.size();}
    IndexCoreset(IndexCoreset &&o) = default;
    IndexCoreset(const IndexCoreset &o) = default;
    void compact(bool shrink_to_fit=true) {
        // TODO: replace with hash map and compact
        std::map<std::pair<IT, FT>, uint32_t> m;
        for(IT i = 0; i < indices_.size(); ++i) {
            ++m[std::make_pair(indices_[i], weights_[i])];
            //++m[std::make_pair(p.first, p.second)];
        }
        if(m.size() == indices_.size()) return;
        size_t newsz = m.size();
        assert(newsz < indices_.size());
        DBG_ONLY(std::fprintf(stderr, "m size: %zu\n", m.size());)
        auto it = &indices_[0];
        auto wit = &weights_[0];
        for(const auto &pair: m) {
            auto [idxw, count] = pair;
            auto [idx, weight] = idxw;
            *it++ = idx;
            *wit++ = count * weight; // Add the weights together
        }
        indices_.resize(newsz);
        weights_.resize(newsz);
        DBG_ONLY(std::fprintf(stderr, "Shrinking to fit\n");)
        if(shrink_to_fit) indices_.shrinkToFit(), weights_.shrinkToFit();
    }
    std::vector<std::pair<IT, FT>> to_pairs() const {
        std::vector<std::pair<IT, FT>> ret(size());
        OMP_PFOR
        for(IT i = 0; i < size(); ++i)
            ret[i].first = indices_[i], ret[i].second = weights_[i];
        return ret;
    }
    void show() {
        for(size_t i = 0, e  = indices_.size(); i < e; ++i) {
            std::fprintf(stderr, "%zu: [%u/%g]\n", i, indices_[i], weights_[i]);
        }
    }
    void resize(size_t newsz) {indices_.resize(newsz); weights_.resize(newsz);}
    IndexCoreset(size_t n): indices_(n), weights_(n) {}
};


template<typename FT=float, typename IT=std::uint32_t>
struct UniformSampler {
    using CoresetType = IndexCoreset<IT, FT>;
    const size_t np_;
    wy::WyRand<IT, 2> rng_;
    UniformSampler(size_t np, uint64_t seed=0): np_(np), rng_(seed) {
    }
    IndexCoreset<IT, FT> sample(const size_t n, uint64_t seed=0) {
        if(seed) rng_.seed(seed);
        IndexCoreset<IT, FT> ret(n);
        for(size_t i = 0; i < n; ++i) {
            ret.indices_[i] = rng_() % np_;
        }
        ret.weights_ = static_cast<FT>(np_) / n; // Ensure final weight = np_
#ifndef NDEBUG
        std::fprintf(stderr, "Weights for uniform: %g (%zu / %zu)\n", static_cast<FT>(np_) / n,
                     np_, n);
#endif
        return ret;
    }
    size_t size() {return np_;}
};

template<typename IT, typename FT, template<typename...> class MapType=flat_hash_map, typename...Extra>
struct MapCoreset {
    using MT = MapType<IT, FT, Extra...>;
    MT data_;
    template<typename...Args>
    MapCoreset(Args &&...args): data_(std::forward<Args>(args)...) {}
    void insert(IT ind, FT w) {
        auto it = data_.find(ind);
        if(it == data_.end())
            data_.emplace(ind, w);
        else
            it->second += w;
    }
};

template<typename IT, typename FT, template<typename...> class MapType=flat_hash_map, typename...Extra>
auto map2index(const MapCoreset<IT, FT, MapType, Extra...> &map) {
    IndexCoreset<IT, FT> ret(map.size());
    size_t ind = 0;
    for(const auto &pair: map)
        ret.indices_[ind] = pair.first, ret.weights_[ind] = pair.second;
    return ret;
}

template<typename FT=float, typename IT=std::uint32_t>
struct CoresetSampler {
    using Sampler = alias::AliasSampler<FT, wy::WyRand<IT, 2>, IT>;
    using CoresetType = IndexCoreset<IT, FT>;
    std::unique_ptr<Sampler>     sampler_;
    std::unique_ptr<FT []>         probs_;
    std::unique_ptr<blz::DV<FT>> weights_;
    std::unique_ptr<blz::DV<IT>> fl_bicriteria_points_; // Used only by FL
    std::unique_ptr<IT []>        fl_asn_;
    size_t                            np_;
    size_t                             k_;
    size_t                             b_;
    uint64_t                  seed_ = 137;
    SensitivityMethod sens_        =  BFL;


    bool ready() const {return sampler_.get();}

    bool operator==(const CoresetSampler &o) const {
        return np_ == o.np_ &&
                       std::equal(probs_.get(), probs_.get() + np_, o.probs_.get()) &&
                      ((weights_.get() == nullptr && o.weights_.get() == nullptr) || // Both are nullptr or
                        std::equal(weights_.get(), weights_.get() + np_, o.weights_.get())); // They're both the same
    }

    std::string to_string() const {
        char buf[256];
        return std::string(buf, std::sprintf(buf, "CoresetSampler<>[np=%zu|k=%zu|b=%zu|SM=%s\n", np_, k_, b_, sm2str(sens_)));
    }

    CoresetSampler(CoresetSampler &&o)      = default;
    CoresetSampler(const CoresetSampler &o) = delete;
    CoresetSampler(std::string s) {
        gzFile ifp = gzopen(s.data(), "rb");
        this->read(ifp);
        gzclose(ifp);
    }
    CoresetSampler() {}

    void write(const std::string &s) const {
        gzFile fp = gzopen(s.data(), "wb");
        if(!fp) throw std::runtime_error("Failed to open file");
        this->write(fp);
        gzclose(fp);
    }
    void read(const std::string &s) {
        gzFile fp = gzopen(s.data(), "rb");
        if(!fp) throw std::runtime_error("Failed to open file");
        this->read(fp);
        gzclose(fp);
    }
    void write(gzFile fp) const {
        uint64_t n = np_;
        gzwrite(fp, &n, sizeof(n));
#if VERBOSE_AF
        std::fprintf(stderr, "Writing %zu\n", size_t(n));
#endif
        gzwrite(fp, &seed_, sizeof(seed_));
        gzwrite(fp, &probs_[0], sizeof(probs_[0]) * np_);
        uint32_t weights_present = weights_ ? 137: 0;
        gzwrite(fp, &weights_present, sizeof(weights_present));
        if(weights_)
            gzwrite(fp, weights_->data(), sizeof(weights_->operator[](0)) * np_);
    }
    void write(std::FILE *fp) const {
        auto fd = ::fileno(fp);
        uint64_t n = np_;
        checked_posix_write(fd, &n, sizeof(n));
        checked_posix_write(fd, &seed_, sizeof(seed_));
        checked_posix_write(fd, &probs_[0], sizeof(probs_[0]) * np_);
        uint32_t weights_present = weights_ ? 137: 0;
        checked_posix_write(fd, &weights_present, sizeof(weights_present));
        if(weights_)
            checked_posix_write(fd, weights_->data(), sizeof(weights_->operator[](0)) * np_);
    }
    void read(gzFile fp) {
        uint64_t n;
        gzread(fp, &n, sizeof(n));
#if VERBOSE_AF
        std::fprintf(stderr, "Reading %zu\n", size_t(n));
#endif
        np_ = n;
        gzread(fp, &seed_, sizeof(seed_));
        probs_.reset(new FT[n]);
        gzread(fp, &probs_[0], sizeof(FT) * n);
        uint32_t weights_present;
        gzread(fp, &weights_present, sizeof(weights_present));
        if(weights_present) {
            assert(weights_present == 137);
            weights_.reset(new blz::DV<FT>(n));
            gzread(fp, weights_->data(), sizeof(FT) * n);
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
            assert(weights_present == 137);
            weights_.reset(new blz::DV<FT>(n));
            ::read(fd, weights_->data(), sizeof(FT) * n);
        }
        sampler_.reset(new Sampler(probs_.get(), probs_.get() + n, seed_));
    }

    template<typename CFT>
    void make_gmm_sampler(size_t ncenters,
                      const CFT *costs, const IT *assignments,
                      uint64_t seed=137,
                      double alpha_est=0.)
    {
        // Note: this takes actual distances and then squares them.
        // ensure that the costs provided are L2Norm, not sqrL2Norm.
        // From Training Gaussian Mixture Models at Scale via Coresets
        // http://www.jmlr.org/papers/volume18/15-506/15-506.pdf
        // Note: this can be expanded to general probability measures.
        throw std::runtime_error("I'm not certain this is correct. Do not use this until I am.");
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
        for(size_t i = 0; i < np_; ++i) {
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
            OMP_ATOMIC
            weighted_cost_sums[asn] += wcost;
            OMP_ATOMIC
            weight_sums[asn] += w; // If unweighted, weights are 1.
            OMP_ATOMIC
            ++center_counts[asn];
            total_cost += wcost;
            sqcosts[i] = cost;
        }
        OMP_PFOR
        for(size_t i = 0; i < np_; ++i) {
            this->probs_[i] = alpha_est * getweight(i) * (sqcosts[i] + weighted_cost_sums[assignments[i]] / weight_sums[assignments[i]])
                        + 2. * total_cost / weight_sums[assignments[i]];
        }
        auto si = 1. / std::accumulate(&this->probs_[0], this->probs_.get() + np_, 0.);
        OMP_PFOR
        for(size_t i = 0; i < np_; ++i)
            this->probs_[i] *= si;
        sampler_.reset(new Sampler(probs_.get(), probs_.get() + np_, seed));
    }
    template<typename CFT>
    void make_sampler(size_t np, size_t ncenters,
                      const CFT *costs, const IT *assignments,
                      const FT *weights=nullptr,
                      uint64_t seed=137,
                      SensitivityMethod sens=BRAVERMAN_FELDMAN_LANG,
                      unsigned k = 0,
                      const IT *centerids = nullptr, // Necessary for FL sampling, otherwise useless
                      double alpha_est=0.)
    {
        sens_ = sens;
        np_ = np;
        b_ = ncenters;
        if(!k) k = ncenters;
        k_ = k;
        if(weights) {
            weights_.reset(new blz::DV<FT>(np_));
            std::memcpy(weights_->data(), weights, sizeof(FT) * np_);
        } else weights_.release();
        if(sens == LUCIC_FAULKNER_KRAUSE_FELDMAN) {
            make_gmm_sampler(ncenters, costs, assignments, seed, alpha_est);
        } else if(sens == VARADARAJAN_XIAO) {
            make_sampler_vx(ncenters, costs, assignments, seed);
        } else if(sens == BFL) {
            make_sampler_bfl(ncenters, costs, assignments, seed);
        } else if(sens == FL) {
            make_sampler_fl(ncenters, costs, assignments, seed, centerids);
        } else if(sens == LBK) {
            make_sampler_lbk(ncenters, costs, assignments, seed);
        } else throw std::runtime_error("Invalid SensitivityMethod");
#if 0
        for(unsigned i = 0; i < np; ++i) {
            std::fprintf(stderr, "point %u has prob %g\n", i, probs_[i]);
        }
#endif
    }
    template<typename CFT>
    void make_sampler_vx(size_t ncenters,
                         const CFT *costs, const IT *assignments,
                         uint64_t seed=137)
    {
        auto cv = blaze::CustomVector<CFT, blaze::unaligned, blaze::unpadded>(const_cast<CFT *>(costs), np_);
        double total_cost =
            weights_ ? blaze::dot(*weights_, cv)
                     : blaze::sum(cv);
        probs_.reset(new FT[np_]);
        blz::CustomVector<FT, blaze::unaligned, blaze::unpadded> sensitivies(probs_.get(), np_);
        std::vector<IT> center_counts(ncenters);
        OMP_PFOR
        for(size_t i = 0; i < np_; ++i) {
            OMP_ATOMIC
            ++center_counts[assignments[i]];
        }
        if(weights_) {
            sensitivies = (*weights_) * cv * (1. / total_cost);
        } else {
            sensitivies = cv * (1. / total_cost);
        }
        // sensitivities = weights * costs / total_cost
        blz::DV<FT> ccinv(ncenters);
        for(unsigned i = 0; i < ncenters; ++i)
            ccinv[i] = 1. / center_counts[i];
        OMP_PFOR
        for(size_t i = 0; i < np_; ++i) {
            sensitivies[i] += ccinv[assignments[i]];
        }
        // sensitivities = weights * costs / total_cost + 1. / (cluster_size)
        const double total_sensitivity = blaze::sum(sensitivies);
        // probabilities = sensitivity / sum(sensitivities) [use the same location in memory because we no longer need sensitivities]
        sensitivies *= 1. / total_sensitivity;
        sampler_.reset(new Sampler(probs_.get(), probs_.get() + np_, seed));
    }
    template<typename CFT>
    void make_sampler_fl(size_t,
                         const CFT *costs, const IT *asn,
                         uint64_t seed=137, const IT *bicriteria_centers=nullptr)
    {
        // See https://arxiv.org/pdf/1106.1379.pdf, figures 2,3,4
        fl_asn_.reset(new IT[np_]);
        blaze::CustomVector<IT, blaze::unaligned, blaze::unpadded>(fl_asn_.get(), np_) =
            blaze::CustomVector<const IT, blaze::unaligned, blaze::unpadded>(asn, np_);
        if(bicriteria_centers) {
            if(!fl_bicriteria_points_) fl_bicriteria_points_.reset(new blz::DV<IT>(b_));
            else fl_bicriteria_points_->resize(b_);
            *fl_bicriteria_points_ = blaze::CustomVector<const IT, blaze::unaligned, blaze::unpadded>(bicriteria_centers, b_);
        }
        auto cv = blaze::CustomVector<CFT, blaze::unaligned, blaze::unpadded>(const_cast<CFT *>(costs), np_);
        auto total_cost =
            weights_ ? blaze::dot(*weights_, cv)
                     : blaze::sum(cv);
        probs_.reset(new FT[np_]);
        sampler_.reset(new Sampler(probs_.get(), probs_.get() + np_, seed));
        double total_cost_inv = 1. / (total_cost);
        if(weights_) {
            OMP_PFOR
            for(size_t i = 0; i < np_; ++i) {
                probs_[i] = getweight(i) * (costs[i]) * total_cost_inv;
            }
        } else {
            blaze::CustomVector<CFT, blaze::unaligned, blaze::unpadded> probv(const_cast<CFT *>(probs_.get()), np_);
            probv = blz::ceil(CFT(np_) * total_cost_inv * cv) + 1.;
        }
        sampler_.reset(new Sampler(probs_.get(), probs_.get() + np_, seed));
    }
    template<typename CFT>
    void make_sampler_lbk(size_t ncenters,
                          const CFT *costs, const IT *assignments,
                          uint64_t seed=137)
    {
        const double alpha = 16 * std::log(k_) + 32., alpha2 = 2. * alpha;

        //auto center_counts = std::make_unique<IT[]>(ncenters);
        blz::DV<FT> weight_sums(ncenters, FT(0));
        blz::DV<FT> cost_sums(ncenters, FT(0));

        double total_costs(0.);
        OMP_PRAGMA("omp parallel for reduction(+:total_costs)")
        for(size_t i = 0; i < np_; ++i) {
            const auto asn = assignments[i];
            assert(asn < ncenters);
            const auto w = getweight(i);

            OMP_ATOMIC
            weight_sums[asn] += w; // If unweighted, weights are 1.
            //OMP_ATOMIC
            //++center_counts[asn];

            const double pointcost = w * costs[i];
            OMP_ATOMIC
            cost_sums[asn] += pointcost;
            total_costs += w * costs[i];
        }
        double weight_sum = blz::sum(weight_sums);
        total_costs /= weight_sum;
        const double tcinv = alpha / total_costs;
        blz::DV<FT> sens(np_);
        for(size_t i = 0; i < ncenters; ++i) {
            cost_sums[i] = alpha2 * cost_sums[i] / (weight_sums[i] * total_costs) + 4 * weight_sum / weight_sums[i];
        }
        OMP_PFOR
        for(size_t i = 0; i < np_; ++i) {
            sens[i] = tcinv * costs[i] + cost_sums[assignments[i]];
        }
        sampler_.reset(new Sampler(sens.data(), sens.data() + np_, seed));
    }
    template<typename CFT>
    void make_sampler_bfl(size_t ncenters,
                          const CFT *costs, const IT *assignments,
                          uint64_t seed=137)
    {
        // This is for a bicriteria approximation
        // Use make_sampler_vx for a constant approximation for arbitrary metric spaces,
        // and make_sampler_lbk for bicriteria approximations for \mu-similar divergences.
        auto cv = blaze::CustomVector<CFT, blaze::unaligned, blaze::unpadded>(const_cast<CFT *>(costs), np_);
        double total_cost =
            weights_ ? blaze::dot(*weights_, cv)
                    : blaze::sum(cv);
        probs_.reset(new FT[np_]);
        double total_probs = 0.;
        std::vector<IT> center_counts(ncenters);
        std::vector<FT> weight_sums(ncenters);
        OMP_PFOR
        for(size_t i = 0; i < np_; ++i) {
            const auto asn = assignments[i];
            assert(asn < ncenters);
            const auto w = getweight(i);
            OMP_ATOMIC
            weight_sums[asn] += w; // If unweighted, weights are 1.
            OMP_ATOMIC
            ++center_counts[asn];
        }
        OMP_PRAGMA("omp parallel for reduction(+:total_probs)")
        for(size_t i = 0; i < np_; ++i) {
            const auto w = getweight(i);
            double fraccost = w * costs[i] / total_cost;
            const auto asn = assignments[i];
            double fracw = w / (weight_sums[asn] * center_counts[asn]);
            probs_[i] = .5 * (fraccost + fracw);
            total_probs += probs_[i];
        }
        // Because this doesn't necessarily sum to 1.
        blaze::CustomVector<FT, blaze::unaligned, blaze::unpadded>(probs_.get(), np_) /= total_probs;
        sampler_.reset(new Sampler(probs_.get(), probs_.get() + np_, seed));
    }
    auto getweight(size_t ind) const {
        return weights_ ? weights_->operator[](ind): static_cast<FT>(1.);
    }
    IndexCoreset<IT, FT> sample(const size_t n, uint64_t seed=0, double eps=0.1) {
        if(unlikely(!sampler_.get())) throw std::runtime_error("Sampler not constructed");
        if(seed) sampler_->seed(seed);
        IndexCoreset<IT, FT> ret(n);
        const double dn = n;
        for(size_t i = 0; i < n; ++i) {
            const auto ind = sampler_->sample();
            assert(ind < np_);
            ret.indices_[i] = ind;
            ret.weights_[i] = getweight(ind) / (dn * probs_[ind]);
        }
        if(sens_ == FL && fl_bicriteria_points_) {
            assert(fl_bicriteria_points_->size() == b_);
            std::unique_ptr<FT[]> wsums(new FT[b_]());
            auto &bicp = *fl_bicriteria_points_;
            for(size_t i = 0; i < n; ++i)
                wsums[fl_asn_[i]] += ret.weights_[i];
            const double wmul = (1. + 10. * eps) * b_;
            ret.resize(n + b_);
            for(size_t i = n; i < ret.size(); ++i) {
                ret.indices_[i] = bicp[i - n];
                ret.weights_[i] = std::max(wmul - wsums[i - n], 0.);
            }
        }
        return ret;
    }
    size_t size() const {return np_;}
};


}//coresets

}// namespace minocore

namespace cs = minocore::coresets;

#endif /* FGC_CORESETS_H__ */
