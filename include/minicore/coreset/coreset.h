#pragma once
#ifndef FGC_CORESETS_H__
#define FGC_CORESETS_H__
#include <vector>
#include <map>
#include <queue>
#include "alias_sampler/alias_sampler.h"
#include "minicore/util/shared.h"
#include "minicore/util/blaze_adaptor.h"
#include <zlib.h>
#include "libsimdsampling/simdsampling.h"
#ifdef _OPENMP
#  include <omp.h>
#endif
//#include "minicore/dist/distance.h"


namespace minicore {
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

static constexpr const SensitivityMethod CORESET_CONSTRUCTIONS [] {
    BRAVERMAN_FELDMAN_LANG,
    FELDMAN_LANGBERG,
    LUCIC_FAULKNER_KRAUSE_FELDMAN,
    VARADARAJAN_XIAO,
    LUCIC_BACHEM_KRAUSE
};

static constexpr const char *sm2str(SensitivityMethod sm) {
    switch(sm) {
        case BFL:  return "BFL";
        case VX:   return "VX";
        case LFKF: return "LFKF";
        case LBK:  return "LBK";
        case FL:   return "FL";
    }
    return "UNKNOWN";
}

static constexpr SensitivityMethod str2sm(const char *s) {
    for(const auto sm: CORESET_CONSTRUCTIONS) if(std::strcmp(sm2str(sm), s) == 0) return sm;
    return BRAVERMAN_FELDMAN_LANG;
}
static inline SensitivityMethod str2sm(const std::string &s) {
    return str2sm(s.data());
}

using namespace std::literals;

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
    IndexCoreset(std::FILE *fp) {this->read(fp);}
    IndexCoreset(gzFile fp) {this->read(fp);}

    void read(gzFile fp) {
        uint64_t sz;
        if(gzread(fp, &sz, sizeof(sz)) != ssize_t(sizeof(sz)))
            goto fail;
        indices_.resize(sz);
        weights_.resize(sz);
        if(gzread(fp, indices_.data(), indices_.size() * sizeof(IT)) != int64_t(indices_.size() * sizeof(indices_[0])))
            goto fail;
        if(gzread(fp, weights_.data(), weights_.size() * sizeof(IT)) != int64_t(weights_.size() * sizeof(weights_[0])))
            goto fail;
        return;
        fail:
            throw std::runtime_error("Failed to read from file");
    }

    void write(gzFile fp) const {
        uint64_t n = size();
        if(gzwrite(fp, &n, sizeof(n)) != sizeof(n)) goto fail;
        if(gzwrite(fp, indices_.data(), indices_.size() * sizeof(IT)) != int64_t(indices_.size() * sizeof(IT)))
            goto fail;
        if(gzwrite(fp, weights_.data(), weights_.size() * sizeof(FT)) != int64_t(weights_.size() * sizeof(FT)))
            goto fail;
        return;
        fail:
            throw std::runtime_error("Failed to write in "s + __PRETTY_FUNCTION__);
    }
    void write(std::FILE *fp) const {
        uint64_t n = size();
        if(std::fwrite(&n, sizeof(n), 1, fp) != 1) goto fail;
        if(std::fwrite(indices_.data(), sizeof(IT), indices_.size(), fp) != indices_.size())
            goto fail;
        if(std::fwrite(weights_.data(), sizeof(FT), weights_.size(), fp) != weights_.size())
            goto fail;
        return;
        fail:
            throw std::runtime_error("Failed to write in "s + __PRETTY_FUNCTION__);
    }
    void write(std::string path) const {
        gzFile fp = gzopen(path.data(), "rb");
        if(!fp) throw std::runtime_error("Failed to open file in "s + __PRETTY_FUNCTION__);
        write(fp);
        gzclose(fp);
    }

    auto &compact(bool shrink_to_fit=true) {
        // TODO: replace with hash map and compact
        std::map<std::pair<IT, FT>, uint32_t> m;
        for(IT i = 0; i < indices_.size(); ++i) {
            ++m[std::make_pair(indices_[i], weights_[i])];
            //++m[std::make_pair(p.first, p.second)];
        }
        if(m.size() == indices_.size()) return *this;
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
        DBG_ONLY(std::fprintf(stderr, "Shrinking to fit: start at %zu\n", size());)
        if(shrink_to_fit) indices_.shrinkToFit(), weights_.shrinkToFit();
        DBG_ONLY(std::fprintf(stderr, "after shrinking: %zu\n", size());)
        return *this;
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
    std::unique_ptr<blaze::DynamicVector<FT>> weights_;
    std::unique_ptr<blaze::DynamicVector<IT>> fl_bicriteria_points_; // Used only by FL
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
        if(fl_asn_ || fl_bicriteria_points_) throw std::runtime_error("Not implemented: serialization for FL coreset samplers");
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
            weights_.reset(new blaze::DynamicVector<FT>(n));
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
            weights_.reset(new blaze::DynamicVector<FT>(n));
            ::read(fd, weights_->data(), sizeof(FT) * n);
        }
        sampler_.reset(new Sampler(probs_.get(), probs_.get() + n, seed_));
    }

    template<typename CFT>
    void make_gmm_sampler(size_t ncenters,
                      const CFT *costs, const IT *assignments,
                      double alpha_est=0.)
    {
        // Note: this takes actual distances and then squares them.
        // ensure that the costs provided are L2Norm, not sqrL2Norm.
        // From Training Gaussian Mixture Models at Scale via Coresets
        // http://www.jmlr.org/papers/volume18/15-506/15-506.pdf
        // Note: this can be expanded to general probability measures.
        std::vector<FT> weight_sums(ncenters), weighted_cost_sums(ncenters);
        std::vector<FT> sqcosts(ncenters);
        std::vector<IT> center_counts(ncenters);
        double total_cost = 0.;

        OMP_PRAGMA("omp parallel for reduction(+:total_cost)")
        for(size_t i = 0; i < np_; ++i) {
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
    }
    void make_alias_sampler(uint64_t seed) {
        auto p = probs_.get(), e = p + np_;
        sampler_.reset(new Sampler(p, e, seed));
    }
    template<typename CFT, typename CFT2=CFT, typename IT2=IT>
    void make_sampler(size_t np, size_t ncenters,
                      const CFT *costs, const IT *assignments,
                      const CFT2 *weights=static_cast<CFT2 *>(nullptr),
                      uint64_t seed=137,
                      SensitivityMethod sens=BRAVERMAN_FELDMAN_LANG,
                      unsigned k = unsigned(-1),
                      const IT2 *centerids = static_cast<IT2 *>(nullptr), // Necessary for FL sampling, otherwise useless
                      bool build_alias_sampler=true,
                      double alpha_est=0.)
    {
        sens_ = sens;
        np_ = np;
        b_ = ncenters;
        if(k == (unsigned)-1) k = ncenters;
        k_ = k;
        if(weights) {
            std::fprintf(stderr,"[%s] Copying to weights\n", __PRETTY_FUNCTION__);
            weights_.reset(new blaze::DynamicVector<FT>(np_));
            std::copy(weights, weights + np, weights_->data());
        } else {
            weights_.release();
            assert(!weights_.get());
        }
        if(sens == LUCIC_FAULKNER_KRAUSE_FELDMAN) {
            make_gmm_sampler(ncenters, costs, assignments, alpha_est);
        } else if(sens == VARADARAJAN_XIAO) {
            make_probs_vx(ncenters, costs, assignments);
        } else if(sens == BFL) {
            make_probs_bfl(ncenters, costs, assignments);
        } else if(sens == FL) {
            make_probs_fl(ncenters, costs, assignments, centerids);
        } else if(sens == LBK) {
            make_probs_lbk(ncenters, costs, assignments);
        } else throw std::runtime_error("Invalid SensitivityMethod");
        if(build_alias_sampler) make_alias_sampler(seed);
    }
    template<typename CFT>
    void make_probs_vx(size_t ncenters,
                         const CFT *costs, const IT *assignments)
    {
        const auto cv = blz::make_cv(const_cast<CFT *>(costs), np_);
        double total_cost =
            weights_ ? blaze::dot(*weights_, cv)
                     : blaze::sum(cv);
        probs_.reset(new FT[np_]);
        auto sensitivities = blz::make_cv(probs_.get(), np_);
        std::vector<IT> center_counts(ncenters);
        OMP_PFOR
        for(size_t i = 0; i < np_; ++i) {
            OMP_ATOMIC
            ++center_counts[assignments[i]];
        }
        if(weights_) {
            sensitivities = (*weights_) * cv / total_cost;
        } else {
            sensitivities = cv / total_cost;
        }
        //std::cerr << "Sensitivities: " << trans(sensitivities) << '\n';
        // sensitivities = weights * costs / total_cost
        blaze::DynamicVector<FT> ccinv(ncenters);
        std::transform(center_counts.begin(), center_counts.end(), ccinv.begin(), [](auto x) -> FT {return FT(1) / x;});
        OMP_PFOR
        for(size_t i = 0; i < np_; ++i) {
            sensitivities[i] += ccinv[assignments[i]];
        }
        const double total_sensitivity = blaze::sum(sensitivities);
        // probabilities = sensitivity / sum(sensitivities) [use the same location in memory because we no longer need sensitivities]
        sensitivities *= 1. / total_sensitivity;
    }
    template<typename CFT, typename IT2=IT, typename OIT=IT>
    void make_probs_fl(size_t,
                       const CFT *costs, const IT2 *asn,
                       const OIT *bicriteria_centers=static_cast<OIT *>(nullptr))
    {
        // See https://arxiv.org/pdf/1106.1379.pdf, figures 2,3,4
        fl_asn_.reset(new IT[np_]);
        auto flv = blz::make_cv(fl_asn_.get(), np_);
        auto aiv = blz::make_cv(const_cast<IT2 *>(asn), np_);
        flv = aiv;
        if(bicriteria_centers) {
            if(!fl_bicriteria_points_) fl_bicriteria_points_.reset(new blaze::DynamicVector<IT>(b_));
            else fl_bicriteria_points_->resize(b_);
            *fl_bicriteria_points_ = blz::make_cv(bicriteria_centers, b_);
        }
        auto cv = blz::make_cv(const_cast<CFT *>(costs), np_);
        auto total_cost =
            weights_ ? blaze::dot(*weights_, cv)
                     : blaze::sum(cv);
        probs_.reset(new FT[np_]);
        double total_cost_inv = 1. / (total_cost);
        if(weights_) {
            OMP_PFOR
            for(size_t i = 0; i < np_; ++i) {
                probs_[i] = getweight(i) * (costs[i]) * total_cost_inv;
            }
        } else {
            auto probv(blz::make_cv(const_cast<FT *>(probs_.get()), np_));
            probv = blaze::ceil(FT(np_) * total_cost_inv * cv) + 1.;
        }
    }
    template<typename CFT>
    void make_probs_lbk(size_t ncenters,
                          const CFT *costs, const IT *assignments)
    {
        const double alpha = 16 * std::log(k_) + 32., alpha2 = 2. * alpha;

        VERBOSE_ONLY(std::fprintf(stderr, "alpha: %g\n", alpha);)
        //auto center_counts = std::make_unique<IT[]>(ncenters);
        blaze::DynamicVector<FT> weight_sums(ncenters, FT(0));
        blaze::DynamicVector<FT> cost_sums(ncenters, FT(0));

        double total_costs(0.);
        OMP_PRAGMA("omp parallel for reduction(+:total_costs)")
        for(size_t i = 0; i < np_; ++i) {
            const auto asn = assignments[i];
            assert(asn < ncenters);
            const auto w = getweight(i);

            VERBOSE_ONLY(std::fprintf(stderr, "weight %zu: %g with asn %u\n", i, w, asn);)
            OMP_ATOMIC
            weight_sums[asn] += w; // If unweighted, weights are 1.
            //OMP_ATOMIC
            //++center_counts[asn];

            const double pointcost = w * costs[i];
            OMP_ATOMIC
            cost_sums[asn] += pointcost;
            total_costs += w * costs[i];
        }
        double weight_sum = blaze::sum(weight_sums);
        VERBOSE_ONLY(std::fprintf(stderr, "wsum: %g\n", weight_sum);)
        total_costs /= weight_sum;
        const double tcinv = alpha / total_costs;
        VERBOSE_ONLY(std::fprintf(stderr, "tcinv: %g\n", tcinv);)
        blaze::DynamicVector<FT> sens(np_);
        for(size_t i = 0; i < ncenters; ++i) {
            cost_sums[i] = alpha2 * cost_sums[i] / (weight_sums[i] * total_costs) + 4 * weight_sum / weight_sums[i];
            VERBOSE_ONLY(std::fprintf(stderr, "Adjusted cost: %g\n", cost_sums[i]);)
        }
        OMP_PFOR
        for(size_t i = 0; i < np_; ++i) {
            sens[i] = tcinv * costs[i] + cost_sums[assignments[i]];
        }
        VERBOSE_ONLY(std::fprintf(stderr, "sensitivity sum: %g\n", sum(sens));)
        probs_.reset(new FT[np_]);
        std::transform(sens.data(),  sens.data() + np_, probs_.get(), [tsi=1./sum(sens)](auto x) {return x * tsi;});
    }
    template<typename CFT>
    void make_probs_bfl(size_t ncenters,
                          const CFT *costs, const IT *assignments)
    {
        // This is for a bicriteria approximation
        // Use make_probs_vx for a constant approximation for arbitrary metric spaces,
        // and make_probs_lbk for bicriteria approximations for \mu-similar divergences.
        const auto cv = blz::make_cv(const_cast<CFT *>(costs), np_);
        double total_cost =
            weights_ ? blaze::dot(*weights_, cv)
                    : blaze::sum(cv);
        //std::fprintf(stderr, "total cost: %g (sum of costs w/o weights: %g)\n", total_cost, std::accumulate(costs, costs + np_, 0.));
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
            //std::fprintf(stderr, "weight: %g. wsum: %g\n", w, weight_sums[asn]);
            OMP_ATOMIC
            ++center_counts[asn];
        }
#if 0
        for(size_t i = 0; i < ncenters; ++i) {
            std::fprintf(stderr, "center %zu has total %zu and weight sum %g\n", i, size_t(center_counts[i]), weight_sums[i]);
        }
#endif
        OMP_PRAGMA("omp parallel for reduction(+:total_probs)")
        for(size_t i = 0; i < np_; ++i) {
            const auto w = getweight(i);
            double fraccost = w * costs[i] / total_cost;
            const auto asn = assignments[i];
            double fracw = w / (weight_sums[asn] * center_counts[asn]);
            probs_[i] = .5 * (fraccost + fracw);
            total_probs += probs_[i];
            //std::fprintf(stderr, "fraccost: %g. fracw: %g. Total weight: %g\n", fraccost, fracw, probs_[i]);
        }
        // Because this doesn't necessarily sum to 1.
        blaze::CustomVector<FT, blaze::unaligned, blaze::unpadded>(probs_.get(), np_) /= total_probs;
    }
    auto getweight(size_t ind) const {
        return weights_ ? weights_->operator[](ind): static_cast<FT>(1.);
    }
    struct importance_compare {
        bool operator()(const std::pair<IT, FT> lh, const std::pair<IT, FT> rh) const {
            return lh.second > rh.second;
        }
    };
    struct importance_queue: public std::priority_queue<std::pair<IT, FT>,
                                                        std::vector<std::pair<IT, FT>>,
                                                        importance_compare>
    {
        auto &getc() {return this->c;}
        const auto &getc() const {return this->c;}
    };
    IndexCoreset<IT, FT> top_outliers(const size_t n) {
        importance_queue topk;
        std::pair<IT, FT> cpoint;
        for(size_t i = 0; i < size(); ++i) {
            FT pi = probs_[i];
            if(topk.size() < n) {
                cpoint = {IT(i), pi};
                topk.push(cpoint);
                continue;
            }
            if(topk.top().second < pi) {
                topk.pop();
                cpoint = {IT(i), pi};
                topk.push(cpoint);
            }
        }
        auto container = std::move(topk.getc());
        // Put the most expensive items in front.
        shared::sort(container.begin(), container.end(), importance_compare());
        IndexCoreset<IT, FT> ret(n);
        const double dn = n;
        for(unsigned i = 0; i < n; ++i) {
            auto ind = container[i].first;
            ret.indices_[i] = ind;
            ret.weights_[i] = getweight(ind) / (dn * container[i].second);
        }
    }
    template<typename P, typename=std::enable_if_t<std::is_arithmetic_v<P>> >
    void sample(P *start, P *end) {
        sampler_->sample(start, end);
    }
    IndexCoreset<IT, FT> sample(const size_t n, uint64_t seed=0, double eps=0.1, bool unique=false) {
        IndexCoreset<IT, FT> ret(n);
        sample(ret, seed, eps, unique);
        return ret;
    }
    void sample(IndexCoreset<IT, FT> &ret, uint64_t seed=0, double eps=0.1, bool unique=false) {
        const size_t n = ret.size();
        if(seed && sampler_) sampler_->seed(seed);
        assert(ret.indices_.size() == n);
        assert(ret.weights_.size() == n);
        assert(probs_.get());
        size_t sampled_directly = n;
        if(sens_ == FL) sampled_directly = std::max((long)(n - b_), 0L);
        shared::flat_hash_map<IT, uint32_t> ctr;
        if(sampler_) {
            for(size_t i = 0; (unique ? ctr.size(): i) < sampled_directly; ++i) {
                const auto ind = sampler_->sample();
                assert(ind < np_);
                if(unique) {
                    auto it = ctr.find(ind);
                    if(it != ctr.end()) ++it->second;
                    else ctr.emplace(ind, uint32_t(1));
                } else {
                    ret.indices_[i] = ind;
                    ret.weights_[i] = getweight(ind) / (static_cast<double>(n) * probs_[ind]);
                }
            }
            if(unique) {
                if(ctr.size() < sampled_directly) {
                    auto flpts = (fl_bicriteria_points_ ? fl_bicriteria_points_->size(): size_t(0));
                    ret.resize(ctr.size() + flpts);
                    std::fprintf(stderr, "After compressing %zu samples into unique items, we have only %zu entries, of which %zu are FL sample\n", n, ret.size(), flpts);
                }
                const size_t csz = ctr.size();
                // Copy from map, sort, and set
                using PairT = typename shared::flat_hash_map<IT, uint32_t>::value_type;
                auto space = std::make_unique<PairT[]>(csz);
                std::copy(ctr.begin(), ctr.end(), space.get());
                shared::sort(space.get(), space.get() + csz,
                    [](const PairT &x, const PairT &y)
                    {return std::tie(x.first, x.second) < std::tie(y.first, y.second);}
                );
                for(size_t i = 0; i < csz; ++i) {
                    const auto idx = space[i].first;
                    ret.indices_[i] = idx;
                    ret.weights_[i] = space[i].second * (getweight(idx) / (n * probs_[idx]));
                }
            }
        } else {
            if(!probs_.get()) throw std::runtime_error("probs not generated, cannot sample");
            if(ret.indices_.size() != n) throw std::runtime_error(std::string("Wrong size ret indices ") + std::to_string(n) + " ," + std::to_string(ret.indices_.size()));
            if(!seed) seed = std::rand();
            auto indices = reservoir_simd::sample_k(probs_.get(), np_, n, seed, unique ? SampleFmt::WITH_REPLACEMENT: SampleFmt::NEITHER);
            DBG_ONLY(for(const auto v: indices) if(v > np_) throw std::runtime_error(std::string("index out of bounds") + std::to_string(v));)
            std::copy(indices.begin(), indices.end(), ret.indices_.data());
            std::sort(ret.indices_.begin(), ret.indices_.end());
            std::transform(ret.indices_.begin(), ret.indices_.end(), ret.weights_.begin(), [&](auto idx) {return getweight(idx) / (static_cast<double>(n) * probs_[idx]);});
        }
        if(sens_ == FL && fl_bicriteria_points_) {
            assert(fl_bicriteria_points_->size() == b_);
            std::unique_ptr<FT[]> wsums(new FT[b_]());
            auto &bicp = *fl_bicriteria_points_;
            size_t i;
            for(i = 0; i < sampled_directly; ++i)
                wsums[fl_asn_[i]] += ret.weights_[i];
            const double wmul = (1. + 10. * eps) * b_;
            auto bit = bicp.begin();
            auto wit = wsums.get();
            std::copy(bicp.begin(), bicp.end(), &ret.indices_[i]);
            for(; i < n; ++i) {
                ret.indices_[i] = *bit++;
                ret.weights_[i] = std::max(wmul - *wit++, 0.);
            }
        }
    }
    size_t size() const {return np_;}
};


}//coresets

}// namespace minicore
namespace cs = minicore::coresets;
#endif /* FGC_CORESETS_H__ */
