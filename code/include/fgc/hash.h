#pragma once
#ifndef FGC_HASH_H__
#define FGC_HASH_H__
#include "blaze_adaptor.h"
#include <random>

namespace fgc {

namespace hash {

template<typename FT=double, bool SO=blz::rowMajor>
class JSDLSHasher {
    // See https://papers.nips.cc/paper/9195-locality-sensitive-hashing-for-f-divergences-mutual-information-loss-and-beyond.pdf
    // for the function.
    // Note that this is an LSH for the JS Divergence, not the metric.
    blz::DM<FT, SO> randproj_;
    blz::DV<FT, SO> boffsets_;
public:
    using ElementType = FT;
    static constexpr bool StorageOrder = SO;
    JSDLSHasher(size_t nd, size_t nh, double r, uint64_t seed=0) {
        if(seed == 0) seed = nd * nh + r;
        std::mt19937_64 mt(seed);
        std::normal_distribution<FT> gen;
        randproj_ = blz::generate(nh, nd, [&](size_t, size_t){return gen(mt);}) / r;
        boffsets_ = blz::generate(nh, [&](size_t){return FT(mt()) / mt.max();});
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
    auto dim() const {return randproj_.columns();}
    auto nh()  const {return boffsets_.size();}
};

template<typename FT=double, bool SO=blz::rowMajor>
class S2JSDLSHasher {
    // See S2JSD-LSH: A Locality-Sensitive Hashing Schema for Probability Distributions
    // https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14692
    // for the derivation
    // Note that this is an LSH for the JS Metric, not the JSD.
    blz::DM<FT, SO> randproj_;
    blz::DV<FT, SO> boffsets_;
public:
    using ElementType = FT;
    static constexpr bool StorageOrder = SO;
    S2JSDLSHasher(size_t nd, size_t nh, double w, uint64_t seed=0) {
        if(seed == 0) seed = nd * nh  + w + 1. / w;
        std::mt19937_64 mt(seed);
        std::normal_distribution<FT> gen;
        randproj_ = blz::abs(blz::generate(nh, nd, [&](size_t, size_t){return gen(mt);}) * (4. / (w * w)));
        boffsets_ = blz::generate(nh, [&](size_t){return FT(mt() / 2) / mt.max();}) - 0.5;
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
    auto dim() const {return randproj_.columns();}
    auto nh()  const {return boffsets_.size();}
};

template<typename Hasher, typename IT=::std::uint32_t, typename KT=IT>
struct LSHTable {
    using ElementType = typename Hasher::ElementType;
    const Hasher hasher_;
    std::unique_ptr<shared::flat_hash_map<KT, std::vector<IT>>[]> tables_;
    const unsigned nh_;
    static constexpr bool SO = Hasher::StorageOrder;

private:
    INLINE void insert(unsigned i, KT key, IT id) {
        auto &table = tables_[i];
        auto it = table.find(key);
        if(it == table.end()) it = table.emplace(key, std::vector<IT>{id}).first;
        else                  it->second.push_back(id);
    }
public:

    template<typename...Args>
    LSHTable(Args &&...args): hasher_(std::forward<Args>(args)...),
                              tables_(new shared::flat_hash_map<KT, std::vector<IT>>[hasher_.nh()]),
                              nh_(hasher_.nh())
    {
    }
    void sort() {
        OMP_PRAGMA("omp parallel for schedule(dynamic)")
        for(unsigned i = 0; i < tables_.size(); ++i)
            for(auto &pair: tables_[i])
                shared::sort(pair.second.begin(), pair.second.end());
    }
    template<typename Query>
    decltype(auto) hash(const Query &q) const {
        return hasher_.hash(q);
    }
    template<typename VT, bool OSO>
    void add(const blz::Vector<VT, OSO> &input, IT id) {
        auto hv = blz::evaluate(hash(input));
        if(nh_ != hv.size()) {
            std::fprintf(stderr, "[%s] nh_: %u. hv.size: %zu\n", __PRETTY_FUNCTION__, nh_, hv.size());
            std::exit(1);
        }
        for(unsigned i = 0; i < nh_; ++i) {
            insert(i, hv[i], id);
        }
    }
    template<typename MT, bool OSO>
    void add(const blz::Matrix<MT, OSO> &input, IT idoffset=0) {
        auto hv = blz::evaluate(trans(hash(input)));
        std::fprintf(stderr, "hv shape: %zu/%zu.\n", hv.rows(), hv.columns());
        if(nh_ != hv.rows()) {
            std::fprintf(stderr, "[%s] nh_: %u. hv.rows: %zu\n", __PRETTY_FUNCTION__, nh_, hv.rows());
            std::exit(1);
        }
        if((~input).rows() != hv.columns()) {
            std::fprintf(stderr, "[%s] nh_: %u. hv.columns: %zu. input rows: %zu\n", __PRETTY_FUNCTION__, nh_, hv.columns(), (~input).rows());
            std::exit(1);
        }
        // Each thread adds to its own hash table to avoid any contention.
        OMP_PFOR
        for(unsigned i = 0; i < nh_; ++i) {
            auto r = row(hv, i BLAZE_CHECK_DEBUG);
            assert(r.size() == (~input).rows());
            SK_UNROLL_4
            for(IT roff = 0, eoff = r.size(); roff < eoff; ++roff) {
                std::fprintf(stderr, "hash table %u is getting key %u at index %u\n", i, unsigned(r[roff]), idoffset + roff);
                insert(i, r[roff], idoffset + roff);
            }
        }
    }
    template<typename VT, bool OSO>
    shared::flat_hash_map<IT, unsigned> query(const blz::Vector<VT, OSO> &query) const {
        auto hv = evaluate(hash(query));
        shared::flat_hash_map<IT, unsigned> ret;
        for(unsigned i = 0; i < nh_; ++i) {
            auto it = tables_[i].find(hv[i]);
            if(it != tables_[i].end()) {
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
        if(hv.rows() != (~query).rows()) throw std::runtime_error("WRong number of rows");
        std::vector<shared::flat_hash_map<IT, unsigned>> ret(hv.rows());
        assert(hv.rows() == (~query).rows());
        OMP_PFOR
        for(unsigned j = 0; j < hv.rows(); ++j) {
            auto &map = ret[j];
            typename shared::flat_hash_map<IT, std::vector<IT>>::const_iterator it;
            auto hr = row(hv, j BLAZE_CHECK_DEBUG);
            assert(hr.size() == nh_);
            for(unsigned i = 0; i < nh_; ++i) {
                if((it = tables_[i].find(KT(hr[i]))) != tables_[i].cend()) {
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
