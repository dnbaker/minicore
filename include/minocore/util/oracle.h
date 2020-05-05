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
            return v & 0xFFFFFFFFu;
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
        return map_.find(!symmetric && lh > rh ? KeyType::make_key(rh, lh): KeyType::make_key(lh, rh));
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

struct MatrixLookup {};

template<typename Mat>
struct MatrixMetric {
    /*
     *  This calculate the distance between item i and item j in this problem
     *  by simply indexing the given array.
     *  This requires precalculation of the array (and space) but saves computation.
     *  By convention, use row index = facility, column index = point
     */
    const Mat &mat_;
    MatrixMetric(const Mat &mat): mat_(mat) {}
    auto operator()(size_t i, size_t j) const {
        return mat_(i, j);
    }
};

template<typename Mat, typename Dist>
struct MatrixDistMetric {
    /*
     *  This calculate the distance between item i and item j in this problem
     *  by calculating the distances between row i and row j under the given distance metric.
     *  This requires precalculation of the array (and space) but saves computation.
     *
     */
    const Mat &mat_;
    const Dist dist_;

    MatrixDistMetric(const Mat &mat, Dist dist): mat_(mat), dist_(std::move(dist)) {}

    auto operator()(size_t i, size_t j) const {
        return dist_(row(mat_, i, blaze::unchecked), row(mat_, j, blaze::unchecked));
    }
};

template<typename Iter, typename Dist>
struct IndexDistMetric {
    /*
     * Adapts random access iterator to use norms between dereferenced quantities.
     */
    const Iter iter_;
    const Dist &dist_;

    IndexDistMetric(const Iter iter, const Dist &dist): iter_(iter), dist_(std::move(dist)) {}

    auto operator()(size_t i, size_t j) const {
        return dist_(iter_[i], iter_[j]);
    }
};

template<typename Iter>
struct BaseOperand {
    using DerefType = decltype((*std::declval<Iter>()));
    using TwiceDerefedType = std::remove_reference_t<decltype(std::declval<DerefType>().operand())>;
    using type = TwiceDerefedType;
};


template<typename Iter>
struct IndexDistMetric<Iter, MatrixLookup> {
    using Operand = typename BaseOperand<Iter>::type;
    using ET = typename Operand::ElementType;
    /* Specialization of above for MatrixLookup
     *
     *
     */
    using Dist = MatrixLookup;
    const Operand &mat_;
    const Dist dist_;

    IndexDistMetric(const Iter iter, Dist dist): mat_((*iter).operand()), dist_(std::move(dist)) {}

    ET operator()(size_t i, size_t j) const {
        assert(i < mat_.rows());
        assert(j < mat_.columns());
        return mat_(i, j);
    }
};

template<typename Iter, typename Dist>
auto make_index_dm(const Iter iter, const Dist &dist) {
    return IndexDistMetric<Iter, Dist>(iter, dist);
}
template<typename Mat, typename Dist>
auto make_matrix_dm(const Mat &mat, const Dist &dist) {
    return MatrixDistMetric<Mat, Dist>(mat, dist);
}
template<typename Mat>
auto make_matrix_m(const Mat &mat) {
    return MatrixMetric<Mat>(mat);
}


template<typename Oracle, template<typename...> class Map=std::unordered_map, bool symmetric=true, bool threadsafe=false, typename IT=std::uint32_t, typename FT=float,
          bool use_row_vector=true>
struct RowCachingOracleWrapper {
    using output_type = std::decay_t<decltype(std::declval<Oracle>()(0,0))>;
    using VType = blaze::DynamicVector<FT, use_row_vector ? blaze::rowVector: blaze::columnVector>;
    using map_type = Map<IT, VType>;
    const Oracle &oracle_;
    mutable map_type map_;
    size_t np_;
private:
    mutable std::shared_mutex mut_;
    using map_iterator = typename map_type::iterator;
    // TODO: use two kinds of locks
public:
    RowCachingOracleWrapper(const Oracle &oracle, size_t np, size_t rsvsz=0): oracle_(oracle), np_(np) {
        map_.reserve(rsvsz ? rsvsz: np);
    }
    template<typename It>
    void cache_range(It start, It end) const {
        unsigned n = std::distance(start, end);
        for(auto i = 0u; i < n; ++i) {
            VType tmp(np_);
            auto lhi = start[i];
            if(map_.find(lhi) != map_.end()) continue;
            OMP_PFOR
            for(size_t j = 0; j < np_; ++j) {
                auto it = map_.find(j);
                tmp[j] = (it == map_.end()) ? oracle_(lhi, j): it->second[lhi];
            }
            map_.emplace(lhi, std::move(tmp));
        }
    }
    output_type operator()(IT lh, IT rh) const {
        std::shared_lock<std::shared_mutex> slock(mut_);
        map_iterator it;
        if((it = map_.find(lh)) != map_.end())
            return it->second[rh];
        if constexpr(symmetric) {
            if((it = map_.find(rh)) != map_.end())
                return it->second[lh];
        }
        VType tmp(np_);
#ifdef _OPENMP
#       pragma omp parallel for
#endif
        for(size_t i = 0; i < np_; ++i) 
            tmp[i] = oracle_(lh, i);
        output_type ret = tmp[rh];
#ifndef NDEBUG
        size_t oldsize = map_.size();
#endif
        if constexpr(threadsafe) {
            slock.unlock();
            std::unique_lock<std::shared_mutex> ulock(mut_);
            if(map_.find(lh) != map_.end()) return ret;
            map_.emplace(lh, std::move(tmp));
        } else {
            map_.emplace(lh, std::move(tmp));
        }
        DBG_ONLY(if(oldsize != map_.size()) std::fprintf(stderr, "New size: %zu\n", map_.size());)
        if constexpr(threadsafe) slock.unlock();
        return ret;
    }
};

template<typename It, typename It2, typename T>
void prep_range(It, It2, const T &) {}

template<typename It, typename It2, typename Oracle, template<typename...> class Map, bool sym, bool ts, typename IT, typename FT, bool use_row_vector>
void prep_range(It start, It2 end, const RowCachingOracleWrapper<Oracle, Map, sym, ts, IT, FT, use_row_vector> &x) {
    x.cache_range(start, end);
}

template<template<typename...> class Map=std::unordered_map, bool symmetric=true, bool threadsafe=false, typename IT=std::uint32_t, typename FT=float, typename Oracle>
auto make_row_caching_oracle_wrapper(const Oracle &oracle, size_t np, size_t rsvsz=0) {
    return RowCachingOracleWrapper<Oracle, Map, symmetric, threadsafe, IT, FT>(oracle, np, rsvsz);
}


} // namespace minocore

#endif /* FGC_ORACLE_H__ */
