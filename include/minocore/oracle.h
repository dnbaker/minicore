#ifndef FGC_ORACLE_H__
#define FGC_ORACLE_H__
#include <vector>
#include <mutex>
#include <unordered_map>

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
    using Type = std::conditional_t<sizeof(IT) == 4, uint64_t, std::pair<IT, IT>>;
    static constexpr Type make_key(IT lh, IT rh) {
        if constexpr(sizeof(IT) == 4) {
            return (uint64_t(lh) << 32) | rh;
        } else {
            return Type{lh, rh};
        }
    }
};

template<typename Oracle, template<typename...> class Map=std::unordered_map, bool symmetric=true, bool threadsafe=false, typename IT=std::uint32_t>
struct CachingOracleWrapper {
    using output_type = std::decay_t<decltype(std::declval<Oracle>()(0,0))>;
    using KeyType = PairKeyType<IT>;
    const Oracle &oracle_;
    Map<typename KeyType::Type, output_type> map_;
    std::mutex mut_;
    CachingOracleWrapper(const Oracle &oracle): oracle_(oracle) {}
    output_type operator()(IT lh, IT rh) {
        if constexpr(symmetric) {
            if(lh > rh) std::swap(lh, rh);
        }
        auto key = make_key(lh, rh);
        output_type ret;
        auto it = map_.find(key);
        if(it == map_.end()) {
            ret = oracle_(lh, rh);
            if constexpr(threadsafe)
            {
                std::lock_guard<std::mutex> mut_;
                map_.emplace(key, ret);
            } else map_.emplace(key, ret);
        } else ret = it->second;
        return ret;
    }
};

template<template<typename...> class Map=std::unordered_map,  bool symmetric=true, bool threadsafe=false, typename IT=std::uint32_t, typename Oracle>
auto make_caching_oracle_wrapper(const Oracle &oracle) {
    return CachingOracleWrapper<Oracle, Map, symmetric, threadsafe, IT>(oracle);
}

} // namespace minocore

#endif /* FGC_ORACLE_H__ */
