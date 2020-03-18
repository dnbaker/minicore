#pragma once
#ifndef FGC_HASH_H__
#define FGC_HASH_H__
#include "blaze_adaptor.h"
#include <random>
#include "xxHash/xxh3.h"
#include "xxHash/xxhash.h"
#ifdef _OPENMP
#  include <mutex>
#endif

namespace fgc {

namespace hash {

struct LSHasherSettings {
    unsigned dim_;
    unsigned k_;
    unsigned l_;
    unsigned nhashes() const {return k_ * l_;}
};

template<typename FT=double, bool SO=blz::rowMajor>
class JSDLSHasher {
    // See https://papers.nips.cc/paper/9195-locality-sensitive-hashing-for-f-divergences-mutual-information-loss-and-beyond.pdf
    // for the function.
    // Note that this is an LSH for the JS Divergence, not the metric.
    blz::DM<FT, SO> randproj_;
    blz::DV<FT, SO> boffsets_;
    LSHasherSettings settings_;
public:
    using ElementType = FT;
    static constexpr bool StorageOrder = SO;
    JSDLSHasher(LSHasherSettings settings, double r, uint64_t seed=0): settings_(settings) {
        unsigned nd = settings.dim_, nh = settings.nhashes();
        if(seed == 0) seed = nd * nh + r;
        std::mt19937_64 mt(seed);
        std::normal_distribution<FT> gen;
        randproj_ = blz::generate(nh, nd, [&](size_t, size_t){return gen(mt);}) / r;
        boffsets_ = blz::generate(nh, [&](size_t){return FT(mt()) / mt.max();});
        assert(settings_.k_ * settings_.l_ == randproj_.rows()); // In case of overflow, I suppose
    }
    template<typename VT>
    decltype(auto) hash(const blz::Vector<VT, SO> &input) const {
        //std::fprintf(stderr, "Regular input size: %zu. my rows/col:%zu/%zu\n", (~input).size(), randproj_.rows(), randproj_.columns());
        return blz::ceil(randproj_ * blz::sqrt(~input) + boffsets_);
    }
    template<typename VT>
    decltype(auto) hash(const blz::Vector<VT, !SO> &input) const {
        //std::fprintf(stderr, "Reversed input size: %zu. my rows/col:%zu/%zu\n", (~input).size(), randproj_.rows(), randproj_.columns());
        return blz::ceil(randproj_ * trans(blz::sqrt(~input)) + boffsets_);
    }
    template<typename VT>
    decltype(auto) hash(const blz::Matrix<VT, SO> &input) const {
        //std::fprintf(stderr, "Regular input rows/col: %zu/%zu. my rows/col:%zu/%zu\n", (~input).rows(), (~input).columns(), randproj_.rows(), randproj_.columns());
        return trans(blz::ceil(randproj_ * trans(~input) + blz::expand(boffsets_, (~input).rows())));
    }
    template<typename VT>
    decltype(auto) hash(const blz::Matrix<VT, !SO> &input) const {
        //std::fprintf(stderr, "Reversed SO input rows/col: %zu/%zu. my rows/col:%zu/%zu\n", (~input).rows(), (~input).columns(), randproj_.rows(), randproj_.columns());
        return trans(blz::ceil(randproj_ * ~input + blz::expand(boffsets_, (~input).columns())));
    }
    const auto &matrix() const {return randproj_;}
    auto dim() const {return randproj_.columns();}
    auto nh()  const {return settings_.nhashes();}
    auto k()   const {return settings_.k_;}
    auto l()   const {return settings_.l_;}
    const auto &settings() const {return settings_;}
};

template<typename FT=double, bool SO=blz::rowMajor>
class S2JSDLSHasher {
    // See S2JSD-LSH: A Locality-Sensitive Hashing Schema for Probability Distributions
    // https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14692
    // for the derivation
    // Note that this is an LSH for the JS Metric, not the JSD.
    blz::DM<FT, SO> randproj_;
    blz::DV<FT, SO> boffsets_;
    LSHasherSettings settings_;
    double w_;
public:
    using ElementType = FT;
    static constexpr bool StorageOrder = SO;
    S2JSDLSHasher(LSHasherSettings settings, double w, uint64_t seed=0): settings_(settings), w_(w) {
        auto nh = settings.nhashes();
        auto nd = settings.dim_;
        if(seed == 0) seed = nd * nh  + w + 1. / w;
        std::mt19937_64 mt(seed);
        std::normal_distribution<FT> gen;
        randproj_ = blz::abs(blz::generate(nh, nd, [&](size_t, size_t){return gen(mt);}) * (4. / (w * w)));
        boffsets_ = blz::generate(nh, [&](size_t){return FT(mt() / 2) / mt.max();}) - 0.5;
        assert(settings_.k_ * settings_.l_ == randproj_.rows()); // In case of overflow, I suppose
    }
    template<typename VT>
    decltype(auto) hash(const blz::Vector<VT, SO> &input) const {
        return blz::floor(blz::sqrt(randproj_ * (~input) + 1.) + boffsets_);
    }
    template<typename VT>
    decltype(auto) hash(const blz::Vector<VT, !SO> &input) const {
        return blz::floor(blz::sqrt(randproj_ * trans(~input) + 1.) + boffsets_);
    }
    template<typename MT>
    decltype(auto) hash(const blz::Matrix<MT, SO> &input) const {
        return trans(blz::floor(blz::sqrt(randproj_ * trans(~input) + 1.) + blz::expand(boffsets_, (~input).rows())));
    }
    template<typename MT>
    decltype(auto) hash(const blz::Matrix<MT, !SO> &input) const {
        return trans(blz::floor(blz::sqrt(randproj_ * (trans(~input)) + 1.) + blz::expand(boffsets_, (~input).columns())));
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
    void add(const blz::Vector<VT, OSO> &input, IT id) {
        auto hv = blz::evaluate(hash(input));
        if(unlikely(nh_ != hv.size())) {
            std::fprintf(stderr, "[%s] nh_: %u. hv.size: %zu\n", __PRETTY_FUNCTION__, nh_, hv.size());
            std::exit(1);
        }
        auto st = settings();
        for(size_t i = 0; i < st.l_; ++i) {
            auto hh = xxhasher_(&hv[i * st.k_], sizeof(ElementType) * st.k_);
            insert(i, hh, id);
        }
    }
    template<typename MT, bool OSO>
    void add(const blz::Matrix<MT, OSO> &input, IT idoffset=0) {
        auto hv = blz::evaluate(hash(input));
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
        for(unsigned i = 0; i < nr; ++i) {
            auto r = row(hv, i, blz::unchecked);
            for(unsigned j = 0; j < _l; ++j) {
                insert(j, xxhasher_(&r[j * _k], sizeof(ElementType) * _k), idoffset + i);
            }
        }
    }
    template<typename VT, bool OSO>
    shared::flat_hash_map<IT, unsigned> query(const blz::Vector<VT, OSO> &query) const {
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
    query(const blz::Matrix<MT, OSO> &query) const {
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

using hash::JSDLSHasher;
using hash::S2JSDLSHasher;
using hash::LSHTable;

}

#endif /* FGC_HASH_H__ */
