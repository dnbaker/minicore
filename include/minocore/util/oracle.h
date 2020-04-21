#ifndef FGC_ORACLE_H__
#define FGC_ORACLE_H__
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include "./macros.h"

namespace minocore {

template<typename Oracle, typename IT=uint32_t>
struct OracleWrapper {
    const Oracle &oracle_;
    const std::vector<IT> lut_;
public:
    template<typename Container>
    OracleWrapper(const Oracle &oracle, const Container &indices): OracleWrapper(oracle, indices.begin(), indices.end()) {
    }
    template<typename It>
    OracleWrapper(const Oracle &oracle, It start, It end):
        oracle_(oracle), lut_(start, end) {}

    INLINE decltype(auto) operator()(size_t i, size_t j) const {
       return oracle_(lookup(i), lookup(j));
    }

    IT lookup(size_t idx) const {
#ifndef NDEBUG
        return lut_.at(idx);
#else
        return lut_[idx];
#endif
    }
};

template<typename Oracle, typename Container>
auto
make_oracle_wrapper(const Oracle &o, const Container &indices) {
    using IT = std::decay_t<decltype(*indices.begin())>;
    return OracleWrapper<Oracle, IT>(o, indices);
}
template<typename Oracle, typename It>
auto
make_oracle_wrapper(const Oracle &o, It start, It end) {
    using IT = std::decay_t<decltype(*start)>;
    return OracleWrapper<Oracle, IT>(o, start, end);
}

template<typename IT>
struct PairKeyType {
    using Type = std::conditional_t<sizeof(IT) <= 4, uint64_t, std::pair<IT, IT>>;
    static constexpr Type make_key(IT lh, IT rh) {
        if constexpr(sizeof(IT) == 4) {
            return (uint64_t(lh) << 32) | rh;
        } else {
            return Type{lh, rh};
        }
    }
    static auto lh(Type v) {
        if constexpr(sizeof(IT) == 4) {
            return uint32_t(v >> 32);
        } else {
            return v.first;
        }
    }
    static auto rh(Type v) {
        if constexpr(sizeof(IT) == 4) {
            static constexpr Type bitmask = static_cast<IT>((uint64_t(1) << 32) - 1);
            return v & bitmask;
        } else {
            return v.second;
        }
    }
};

template<typename Oracle, template<typename...> class Map=std::unordered_map, bool symmetric=true, bool threadsafe=false, typename IT=std::uint32_t>
struct CachingOracleWrapper {
    using output_type = std::decay_t<decltype(std::declval<Oracle>()(0,0))>;
    using KeyType = PairKeyType<IT>;
    const Oracle &oracle_;
    mutable Map<typename KeyType::Type, output_type> map_;
private:
    mutable std::shared_mutex mut_;
    using iterator_type = decltype(map_.find(std::declval<typename KeyType::Type>()));
    // TODO: use two kinds of locks
public:
    CachingOracleWrapper(const Oracle &oracle): oracle_(oracle) {}
    output_type operator()(IT lh, IT rh) const {
        if constexpr(symmetric) {
            if(lh > rh) std::swap(lh, rh);
        }
        auto key = KeyType::make_key(lh, rh);
        output_type ret;
        {
            std::shared_lock<decltype(mut_)> shared(mut_);
            if(iterator_type it; (it = map_.find(key)) == map_.end()) {
                ret = oracle_(lh, rh);
                if constexpr(threadsafe)
                {
                    shared.unlock();
                    std::unique_lock<decltype(mut_)> lock(mut_);
                    map_.emplace(key, ret);
                } else {
                    map_.emplace(key, ret);
                }
            } else {
                ret = it->second;
                shared.unlock();
            }
        }
        return ret;
    }
    typename Map<typename KeyType::Type, output_type>::iterator
    find(IT lh, IT rh) const {
        return map_.find(is_symmetric || lh < rh ? KeyType::make_key(lh, rh): KeyType::make_key(lh, rh));
    }
    bool contains(IT lh, IT rh) const {
        if constexpr(symmetric) if(lh > rh) std::swap(lh, rh);
        auto key = KeyType::make_key(lh, rh);
        std::shared_lock<decltype(mut_)> shared(mut_);
        return map_.find(key) != map_.end();
    }
};

template<template<typename...> class Map=std::unordered_map,  bool symmetric=true, bool threadsafe=false, typename IT=std::uint32_t, typename Oracle>
auto make_caching_oracle_wrapper(const Oracle &oracle) {
    return CachingOracleWrapper<Oracle, Map, symmetric, threadsafe, IT>(oracle);
}

} // namespace minocore

#endif /* FGC_ORACLE_H__ */
