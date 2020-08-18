#pragma once
#ifndef FGC_HASH_H__
#define FGC_HASH_H__
#include "minocore/util/blaze_adaptor.h"
#include "minocore/util/macros.h"
#include <random>
#include "xxHash/xxh3.h"
#include "xxHash/xxhash.h"
#ifdef _OPENMP
#  include <mutex>
#endif

namespace minocore {

namespace hash {

struct LSHasherSettings {
    unsigned dim_;
    unsigned k_;
    unsigned l_;

    unsigned nhashes() const {return k_ * l_;}
    LSHasherSettings(const LSHasherSettings &) = default;
    LSHasherSettings(LSHasherSettings &&) = default;

    LSHasherSettings(unsigned d, unsigned k, unsigned l): dim_(d), k_(k), l_(l) {}
    LSHasherSettings(std::initializer_list<unsigned> il) {
        if(il.size() != 3) throw std::invalid_argument("LSHasherSettings requires 3 values");
        auto beg = il.begin();
        dim_ = *beg++, k_ = *beg++, l_ = *beg;
    }
};

template<typename FT>
struct clipped_cauchy_distribution {
    const FT max_;
    mutable std::cauchy_distribution<FT> dist_;
    FT min() const {return -max_;}
    FT max() const {return max_;}
    clipped_cauchy_distribution(FT max=1000.): max_(std::abs(max))
    {
    }
    template<typename RNG>
    FT operator()(RNG &x) const {
        FT ret = dist_(x);
        if(ret > max_) ret = max_;
        if(ret < -max_) ret = -max_;
        return ret;
    }
    void reset() {}
};

template<typename FT>
struct cms_distribution {
    // p-stable sampler for p [0,2]
    // Needs testing.
public:
    const FT a_, b_;
private:
    FT t1_, t2_, t2i_, t1sp1_;
    mutable std::uniform_real_distribution<FT> urd_;
    mutable std::exponential_distribution<FT> expd_;
public:
    cms_distribution(double a=2., double b=0): a_(a), b_(b)
    {
        if(a < 0 || a > 2) throw std::out_of_range("a is out of range [0,2]");
        t1_  = -b_ * std::tan(M_PI_2 * a_);
        t2_  = a_ == 1. ? FT(M_PI_2) : std::atan(-t1_) / a_;
        t2i_ = 1. / t2_;
        t1sp1_ = std::pow(1. + t1_ * t1_, .5 / a_);
    }
    template<typename RNG>
    FT operator()(RNG &rng) const {
        auto U = (urd_(rng) - .5) * M_PI, W = expd_(rng);
        FT s;
        if(a_ == 1.) {
            const auto bu = b_ * U;
            s = t2i_ * ((M_PI_2 + bu) * std::tan(U) - b_ * std::log(M_PI_2 * W * std::cos(U) / (M_PI_2 + bu)));
        } else {
            const auto ainv = 1. / a_;
            const auto at2u = a_ * (t2_ + U);
            s = t1sp1_
                * std::sin(at2u) / std::pow(std::cos(U), ainv)
                * std::pow(std::cos(U - at2u) / W, (1. - a_) * ainv);
        }
        return s;
    }
};

template<typename FT=double, bool SO=blaze::rowMajor>
class JSDLSHasher {
    // See https://papers.nips.cc/paper/9195-locality-sensitive-hashing-for-f-divergences-mutual-information-loss-and-beyond.pdf
    // for the function.
    // Note that this is an LSH for the JS Divergence, not the metric.
    //
    // This relies on a U/H-approximation of the JSD (capacitory discrimination in Topsoe, 2000)
    // by the Hellinger distance, and uses an LSH for the Hellinger as-is.
    blaze::DynamicMatrix<FT, SO> randproj_;
    blaze::DynamicVector<FT, SO> boffsets_;
    LSHasherSettings settings_;
public:
    using ElementType = FT;
    static constexpr bool StorageOrder = SO;
    static constexpr const FT rngnorm = 1.0/(1ull<<52);
    JSDLSHasher(LSHasherSettings settings, const double r, uint64_t seed=0): settings_(settings) {
        unsigned nd = settings.dim_, nh = settings.nhashes();
        if(seed == 0) seed = nd * nh + r;
        seed |= 1; // Ensures that the seed is odd, necessary
        std::normal_distribution<FT> gen;
        randproj_ = blaze::generate(nh, nd, [seed,&gen](uint64_t x, size_t y) ALWAYS_INLINE {
            wy::WyRand<uint64_t> rng(((x << 32) | y) ^ seed);
            return gen(rng);
        });
        randproj_ /= r;
        boffsets_.resize(nh);
        std::mt19937_64 mt(seed);
        for(auto &i: boffsets_) i = FT(mt()) / mt.max();
        assert(settings_.k_ * settings_.l_ == randproj_.rows()); // In case of overflow
    }
    JSDLSHasher(unsigned dim, unsigned k, unsigned l, const double r, uint64_t seed=0): JSDLSHasher(LSHasherSettings{dim, k, l}, r, seed)
    {}
    template<typename VT>
    decltype(auto) hash(const blaze::Vector<VT, SO> &input) const {
        //std::fprintf(stderr, "Regular input size: %zu. my rows/col:%zu/%zu\n", (~input).size(), randproj_.rows(), randproj_.columns());
        return randproj_ * blaze::sqrt(~input) + boffsets_;
    }
    template<typename VT>
    decltype(auto) hash(const blaze::Vector<VT, !SO> &input) const {
        //std::fprintf(stderr, "Reversed input size: %zu. my rows/col:%zu/%zu\n", (~input).size(), randproj_.rows(), randproj_.columns());
        return randproj_ * trans(blaze::sqrt(~input)) + boffsets_;
    }
    template<typename VT>
    decltype(auto) hash(const blaze::Matrix<VT, SO> &input) const {
        //std::fprintf(stderr, "Regular input rows/col: %zu/%zu. my rows/col:%zu/%zu\n", (~input).rows(), (~input).columns(), randproj_.rows(), randproj_.columns());
        return trans(randproj_ * trans(blaze::sqrt(~input)) + blaze::expand(boffsets_, (~input).rows()));
    }
    template<typename VT>
    decltype(auto) hash(const blaze::Matrix<VT, !SO> &input) const {
        //std::fprintf(stderr, "Reversed SO input rows/col: %zu/%zu. my rows/col:%zu/%zu\n", (~input).rows(), (~input).columns(), randproj_.rows(), randproj_.columns());
        return trans(randproj_ * blaze::sqrt(~input) + blaze::expand(boffsets_, (~input).columns()));
    }
    template<typename...Args>
    decltype(auto) hash(Args &&...args) const {
        return ceil(project(std::forward<Args>(args)...));
    }
    const auto &matrix() const {return randproj_;}
    auto dim() const {return randproj_.columns();}
    auto nh()  const {return settings_.nhashes();}
    auto k()   const {return settings_.k_;}
    auto l()   const {return settings_.l_;}
    const auto &settings() const {return settings_;}
};

template<typename FT=double, bool SO=blaze::rowMajor>
class HellingerLSHasher: public JSDLSHasher<FT, SO> {
public:
    template<typename...Args>
    HellingerLSHasher(Args &&...args): JSDLSHasher<FT, SO>(std::forward<Args>(args)...) {}
};


template<template<typename...> class Distribution, typename FT, bool SO, bool use_offsets, typename...Args>
class PStableLSHasher {
    blaze::DynamicMatrix<FT, SO> randproj_;
    blaze::DynamicVector<FT, SO> boffsets_;
    LSHasherSettings settings_;
    double w_;
public:
    using ElementType = FT;
    static constexpr bool StorageOrder = SO;
    template<typename...CArgs>
    PStableLSHasher(unsigned dim, unsigned k, unsigned l, double w, uint64_t seed, CArgs &&...args):
        PStableLSHasher(LSHasherSettings{dim, k, l}, w, seed, std::forward<CArgs>(args)...) {}
    template<typename...CArgs>
    PStableLSHasher(LSHasherSettings settings, double w, uint64_t seed, CArgs &&...args):
        settings_(settings), w_(w)
    {
        Distribution<FT, Args...> gen(std::forward<CArgs>(args)...);
        auto nh = settings.nhashes();
        auto nd = settings.dim_;
        if(seed == 0) seed = nd * nh  + w + 1. / w;
        std::mt19937_64 mt(seed);
        randproj_ = blaze::abs(blaze::generate(nh, nd, [&](size_t, size_t){return gen(mt);}) * (1. / w));
        if constexpr(use_offsets)
            boffsets_ = blaze::generate(nh, [&](size_t){return FT(mt() / 2) / mt.max();}) - 0.5;
        assert(settings_.k_ * settings_.l_ == randproj_.rows()); // In case of overflow, I suppose
    }
    template<typename VT>
    decltype(auto) project(const blaze::Vector<VT, SO> &input) const {
        if constexpr(use_offsets) return randproj_ * (~input) + 1. + boffsets_;
        else                      return randproj_ * (~input);
    }
    template<typename VT>
    decltype(auto) project(const blaze::Vector<VT, !SO> &input) const {
        if constexpr(use_offsets) return randproj_ * trans(~input) + 1. + boffsets_;
        else                      return randproj_ * trans(~input);
    }
    template<typename MT>
    decltype(auto) project(const blaze::Matrix<MT, SO> &input) const {
        if constexpr(use_offsets)
            return trans(randproj_ * trans(~input) + blaze::expand(boffsets_, (~input).rows()));
        else
            return trans(randproj_ * trans(~input));
    }
    template<typename MT>
    decltype(auto) project(const blaze::Matrix<MT, !SO> &input) const {
        if constexpr(use_offsets)
            return trans(randproj_ * trans(~input) + blaze::expand(boffsets_, (~input).columns()));
        else
            return trans(randproj_ * trans(~input));
    }
    template<typename...HArgs>
    decltype(auto) hash(Args &&...args) const {
        return floor(project(std::forward<HArgs>(args)...));
    }
    const auto &matrix() const {return randproj_;}
    auto dim() const {return settings_.dim_;}
    auto nh()  const {return settings_.nhashes();}
    auto k()   const {return settings_.k_;}
    auto l()   const {return settings_.l_;}
    const auto &settings() const {return settings_;}
};

template<typename FT=double, bool SO=blaze::rowMajor, bool use_offsets=true>
class L2LSHasher: public PStableLSHasher<std::normal_distribution, FT, SO, use_offsets> {
public:
    using super = PStableLSHasher<std::normal_distribution, FT, SO, use_offsets>;
    template<typename...Args>
    L2LSHasher(LSHasherSettings settings, double w, uint64_t seed=0, Args &&...args): super(settings, w, seed, std::forward<Args>(args)...)
    {
    }
};

template<typename FT=double, bool SO=blaze::rowMajor, bool use_offsets=true>
class L1LSHasher: public PStableLSHasher<std::cauchy_distribution, FT, SO, use_offsets> {
public:
    using super = PStableLSHasher<std::cauchy_distribution, FT, SO, use_offsets>;
    template<typename...Args>
    L1LSHasher(LSHasherSettings settings, double w, uint64_t seed=0, Args &&...args): super(settings, w, seed, std::forward<Args>(args)...)
    {
    }
};
template<typename FT=double, bool SO=blaze::rowMajor, bool use_offsets=true>
class LpLSHasher: public PStableLSHasher<cms_distribution, FT, SO, use_offsets> {
public:
    using super = PStableLSHasher<cms_distribution, FT, SO, use_offsets>;
    template<typename...Args>
    LpLSHasher(LSHasherSettings settings, double p, double w, uint64_t seed=0): super(settings, w, seed, p)
    {
    }
};

template<typename FT=double, bool SO=blaze::rowMajor, bool use_offsets=true>
class ClippedL1LSHasher: public PStableLSHasher<clipped_cauchy_distribution, FT, SO, use_offsets> {
public:
    using super = PStableLSHasher<clipped_cauchy_distribution, FT, SO, use_offsets>;
    template<typename...Args>
    ClippedL1LSHasher(Args &&...args): super(std::forward<Args>(args)...) {}
};

template<typename FT=double, bool SO=blaze::rowMajor, bool use_offsets=true>
class TVDLSHasher: public L1LSHasher<FT, SO, use_offsets> {
public:
    using super = L1LSHasher<FT, SO, use_offsets>;
    template<typename...Args>
    TVDLSHasher(Args &&...args): super(std::forward<Args>(args)...) {}
};


template<typename FT=double, bool SO=blaze::rowMajor>
class S2JSDLSHasher {
    // See S2JSD-LSH: A Locality-Sensitive Hashing Schema for Probability Distributions
    // https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14692
    // for the derivation
    // Note that this is an LSH for the JS Metric, not the JSD.
    blaze::DynamicMatrix<FT, SO> randproj_;
    blaze::DynamicVector<FT, SO> boffsets_;
    LSHasherSettings settings_;
    double w_;
public:
    using ElementType = FT;
    static constexpr bool StorageOrder = SO;
    S2JSDLSHasher(unsigned dim, unsigned k, unsigned l, const double w, uint64_t seed=0): S2JSDLSHasher(LSHasherSettings{dim, k, l}, w, seed)
    {}
    S2JSDLSHasher(LSHasherSettings settings, double w, uint64_t seed=0): settings_(settings), w_(w) {
        auto nh = settings.nhashes();
        auto nd = settings.dim_;
        if(seed == 0) seed = nd * nh  + w + 1. / w;
        std::mt19937_64 mt(seed);
        std::normal_distribution<FT> gen;
        auto wyseed = mt();
        randproj_ = blaze::abs(blaze::generate(nh, nd, [&](size_t i, size_t j){
            wy::WyHash<uint64_t> wyh(((i << 32) | j) ^ wyseed);
            return gen(wyh);
        }) * (4. / (w * w)));
        wyseed = mt();
        boffsets_ = blaze::generate<SO>(nh, [&](size_t i) {
            return (wy::WyHash<uint64_t>(i ^ wyseed)() >> 12) * (1. / (1ull << 52));
        }) - 0.5;
        assert(settings_.k_ * settings_.l_ == randproj_.rows()); // In case of overflow, I suppose
    }
    template<typename VT>
    decltype(auto) project(const blaze::Vector<VT, SO> &input) const {
        return blaze::sqrt(randproj_ * (~input) + 1.) + boffsets_;
    }
    template<typename VT>
    decltype(auto) project(const blaze::Vector<VT, !SO> &input) const {
        return blaze::sqrt(randproj_ * trans(~input) + 1.) + boffsets_;
    }
    template<typename MT>
    decltype(auto) project(const blaze::Matrix<MT, SO> &input) const {
        return trans(blaze::sqrt(randproj_ * trans(~input) + 1.) + blaze::expand(boffsets_, (~input).rows()));
    }
    template<typename MT>
    decltype(auto) project(const blaze::Matrix<MT, !SO> &input) const {
        return trans(blaze::sqrt(randproj_ * (trans(~input)) + 1.) + blaze::expand(boffsets_, (~input).columns()));
    }
    template<typename...Args>
    decltype(auto) hash(Args &&...args) const {
        return floor(project(std::forward<Args>(args)...));
    }
    const auto &matrix() const {return randproj_;}
    auto dim() const {return settings_.dim_;}
    auto nh()  const {return settings_.nhashes();}
    auto k()   const {return settings_.k_;}
    auto l()   const {return settings_.l_;}
    const auto &settings() const {return settings_;}
};

struct XXHasher32 {
    const uint32_t seed_;
    using type = uint32_t;
    XXHasher32(uint32_t seed=0): seed_(seed) {
    }
    INLINE uint32_t operator()(const void *dat, size_t nb) const {
        return XXH32(dat, nb, seed_);
    }
};
struct XXHasher64 {
    const uint64_t seed_;
    using type = uint64_t;
    XXHasher64(uint64_t seed=0): seed_(seed) {
    }
    INLINE uint64_t operator()(const void *dat, size_t nb) const {
        return XXH3_64bits_withSeed(dat, nb, seed_);
    }
};

template<typename KT> struct XXHasher:
    public std::conditional_t<sizeof(KT) == 8, XXHasher64, XXHasher32>
{
    using super = std::conditional_t<sizeof(KT) == 8, XXHasher64, XXHasher32>;
    XXHasher(typename super::type seed): super(seed) {}
    static_assert(std::is_integral<KT>::value || sizeof(KT) >= 16, "KT must be integral __{u,}int128 aren't guaranteed to have type_traits defined accordingly");
};

template<typename Hasher, typename IT=::std::uint32_t, typename KT=uint64_t>
struct LSHTable {
    using ElementType = typename Hasher::ElementType;
    const Hasher hasher_;
    std::unique_ptr<shared::flat_hash_map<KT, std::vector<IT>>[]> tables_;
    const unsigned nh_;
    XXHasher<KT> xxhasher_;
    OMP_ONLY(std::unique_ptr<std::mutex[]> mutexes;)
    size_t ids_used_ = 0;

    static constexpr bool SO = Hasher::StorageOrder;

private:
    INLINE void insert(unsigned i, KT key, IT id) {
        auto &table = tables_[i];
        {
            OMP_ONLY(std::lock_guard<std::mutex> lock(mutexes[i]);)
            auto it = table.find(key);
            if(it == table.end()) it = table.emplace(key, std::vector<IT>{id}).first;
            else                  it->second.push_back(id);
        }
    }
public:

    template<typename...Args>
    LSHTable(Args &&...args): hasher_(std::forward<Args>(args)...),
                              tables_(new shared::flat_hash_map<KT, std::vector<IT>>[hasher_.l()]),
                              nh_(hasher_.nh()),
                              xxhasher_(XXH3_64bits_withSeed(hasher_.matrix().data(), hasher_.matrix().spacing() *
                                  (blaze::IsRowMajorMatrix_v<
                                       std::decay_t<decltype(hasher_.matrix())>
                                   > ? hasher_.matrix().rows(): hasher_.matrix().columns()),
                                  hasher_.nh()))
                              OMP_ONLY(, mutexes(new std::mutex[hasher_.l()]))
    {
    }
    LSHTable(const LSHTable &) = delete;
    LSHTable(LSHTable &&)     = default;

    void sort() {
        OMP_PRAGMA("omp parallel for schedule(dynamic)")
        for(unsigned i = 0; i < tables_.size(); ++i)
            for(auto &pair: tables_[i])
                shared::sort(pair.second.begin(), pair.second.end());
    }
    const LSHasherSettings &settings() const {return hasher_.settings();}
    auto k()   const {return settings().k_;}
    auto l()   const {return settings().l_;}
    template<typename Query>
    decltype(auto) hash(const Query &q) const {
        return hasher_.hash(q);
    }
    template<typename VT, bool OSO>
    void add(const blaze::Vector<VT, OSO> &input, IT id) {
        auto hv = blaze::evaluate(hash(input));
        if(unlikely(nh_ != hv.size())) {
            std::fprintf(stderr, "[%s] nh_: %u. hv.size: %zu\n", __PRETTY_FUNCTION__, nh_, hv.size());
            std::exit(1);
        }
        auto st = settings();
        for(size_t i = 0; i < st.l_; ++i) {
            auto hh = xxhasher_(&hv[i * st.k_], sizeof(ElementType) * st.k_);
            insert(i, hh, id);
        }
        ++ids_used_;
    }
    template<typename MT, bool OSO>
    void add(const blaze::Matrix<MT, OSO> &input, IT idoffset=0) {
        auto hv = blaze::evaluate(hash(input));
        std::fprintf(stderr, "hv shape: %zu/%zu.\n", hv.rows(), hv.columns());
        if(nh_ != hv.columns()) {
            std::fprintf(stderr, "[%s] nh_: %u. hv.columns: %zu\n", __PRETTY_FUNCTION__, nh_, hv.columns());
            std::exit(1);
        }
        if((~input).rows() != hv.rows()) {
            std::fprintf(stderr, "[%s] nh_: %u. hv.columns: %zu. input rows: %zu\n", __PRETTY_FUNCTION__, nh_, hv.rows(), (~input).rows());
            std::exit(1);
        }
        const size_t nr = (~input).rows();
        const auto _l = l(), _k = k();
        OMP_PFOR
        for(unsigned i = 0; i < nr; ++i) {
            auto r = row(hv, i, blaze::unchecked);
            for(unsigned j = 0; j < _l; ++j) {
                insert(j, xxhasher_(&r[j * _k], sizeof(ElementType) * _k), idoffset + i);
            }
        }
        ids_used_ += nr;
    }
    template<typename VT, bool OSO>
    std::vector<std::pair<IT, unsigned>> topk(const blaze::Vector<VT, OSO> &query, unsigned maxgather=0) const {
        // TODO: build with a heap
        if(!maxgather) maxgather = ids_used_;
        std::vector<std::pair<IT, unsigned>> ret;
        auto retupdate = [&](auto x) {
            auto rit = std::find(ret.begin(), ret.end(), x);
        };
        auto hv = evaluate(hash(query));
        for(unsigned i = 0; i < l(); ++i) {
            if(auto it = tables_[i].find(xxhasher_(&hv[i * k()], sizeof(ElementType) * k()));
               it != tables_[i].end())
            {
                for(const auto v: it->second) {
                    auto rit = std::find_if(ret.begin(), ret.end(), [v](auto x) {return x.first == v;});
                    if(rit == ret.end()) ret.emplace_back({v, 1u});
                    else                 ++rit->second;
                }
            }
        }
        shared::sort(ret.begin(), ret.end(), [](auto x, auto y) {return x.second > y.second;});
        if(maxgather < ret.size()) ret.resize(maxgather);
        return ret;
    }
    template<typename VT, bool OSO>
    shared::flat_hash_map<IT, unsigned> query(const blaze::Vector<VT, OSO> &query) const {
        auto hv = evaluate(hash(query));
        shared::flat_hash_map<IT, unsigned> ret;
        for(unsigned i = 0; i < l(); ++i) {
            if(auto it = tables_[i].find(xxhasher_(&hv[i * k()], sizeof(ElementType) * k()));
               it != tables_[i].end())
            {
                for(const auto v: it->second) {
                    auto nit = ret.find(v);
                    if(nit != ret.end()) ++nit->second;
                    else  ret.emplace(v, 1);
                }
            }
        }
        return ret;
    }
    template<typename MT, bool OSO>
    std::vector<shared::flat_hash_map<IT, unsigned>>
    query(const blaze::Matrix<MT, OSO> &query) const {
        auto hv = evaluate(hash(query));
        //std::fprintf(stderr, "hv rows: %zu. columns: %zu. nh: %u. input num rows: %zu. input col: %zu\n", hv.rows(), hv.columns(), nh_, (~query).rows(), (~query).columns());
        if(hv.columns() != nh_) throw std::runtime_error("Wrong number of columns");
        if(hv.rows() != (~query).rows()) throw std::runtime_error("Wrong number of rows");
        std::vector<shared::flat_hash_map<IT, unsigned>> ret(hv.rows());
        assert(hv.rows() == (~query).rows());
        OMP_PFOR
        for(unsigned j = 0; j < hv.rows(); ++j) {
            auto &map = ret[j];
            typename shared::flat_hash_map<KT, std::vector<IT>>::const_iterator it;
            auto hr = row(hv, j BLAZE_CHECK_DEBUG);
            assert(hr.size() == nh_);
            for(unsigned i = 0; i < l(); ++i) {
                if((it = tables_[i].find(xxhasher_(&hr[i * k()], sizeof(ElementType) * k()))) != tables_[i].cend()) {
                    for(const auto v: it->second) {
                        auto nit = map.find(v);
                        if(nit != map.end()) ++nit->second;
                        else             map.emplace(v, 1);
                    }
                }
            }
        }
        return ret;
    }
};


} // namespace hash

using hash::clipped_cauchy_distribution;
using hash::cms_distribution;

using hash::LSHasherSettings;

using hash::L1LSHasher;        // D_{L1} = \sqrt{\sum_{d=1}^D |P_i - Q_i|}
using hash::L2LSHasher;        // D_{L2} = \sqrt{\sum_{d=1}^D (P_i - Q_i)^2}
using hash::ClippedL1LSHasher;
/* Note:
 *      ClippedL1LSHasher has the same properties as L1, though it is likely to have reduced variance
        and may be more accurate.
        It can be parameterized, but by default clips values to be < 1000 or > -1000.
 */

using hash::JSDLSHasher;       // D_{JS}(P || Q)
using hash::HellingerLSHasher; // D_{Hellinger}(P || Q) = D_{\ell_2}(\sqrt{P}, \sqrt{Q})
using hash::S2JSDLSHasher;     // D_{JSM}(P || Q) = \sqrt{D_{JS}(P || Q)}
using hash::TVDLSHasher;       // D_{TV}(P || Q)  = \frac{D_{\ell_1}(P || Q)}{2}

using hash::LSHTable;

}

#endif /* FGC_HASH_H__ */
