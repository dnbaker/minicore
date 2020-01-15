#ifndef FGC_DISTANCE_AND_MEANING_H__
#define FGC_DISTANCE_AND_MEANING_H__
#include "blaze_adaptor.h"
#include <vector>
#if VERBOSE_AF
#include <iostream>
#endif

namespace blz {

inline namespace distance {
#define DECL_DIST(norm) \
template<typename FT, bool SO>\
INLINE double norm##Dist(const blaze::DynamicVector<FT, SO> &lhs, const blaze::DynamicVector<FT, SO> &rhs) {\
    return norm##Norm(rhs - lhs);\
}\
template<typename FT, bool SO>\
INLINE double norm##Dist(const blaze::DenseVector<FT, SO> &lhs, const blaze::DenseVector<FT, SO> &rhs) {\
    return norm##Norm(~rhs - ~lhs);\
}\
\
template<typename FT, bool SO>\
INLINE double norm##Dist(const blaze::DenseVector<FT, SO> &lhs, const blaze::DenseVector<FT, !SO> &rhs) {\
    return norm##Norm(~rhs - trans(~lhs));\
}\
\
template<typename FT, bool SO>\
INLINE double norm##Dist(const blaze::SparseVector<FT, SO> &lhs, const blaze::SparseVector<FT, SO> &rhs) {\
    return norm##Norm(~rhs - ~lhs);\
}\
\
template<typename FT, bool SO>\
INLINE double norm##Dist(const blaze::SparseVector<FT, SO> &lhs, const blaze::SparseVector<FT, !SO> &rhs) {\
    return norm##Norm(~rhs - trans(~lhs));\
}\
\
template<typename FT, bool AF, bool PF, bool SO, bool OAF, bool OPF>\
INLINE double norm##Dist(const blaze::CustomVector<FT, AF, PF, SO> &lhs,\
                      const blaze::CustomVector<FT, OAF, OPF, SO> &rhs)\
{\
    return norm##Norm(rhs - lhs);\
}\
\
template<typename FT, bool AF, bool PF, bool SO, bool OAF, bool OPF>\
INLINE double norm##Dist(const blaze::CustomVector<FT, AF, PF, SO> &lhs,\
                      const blaze::CustomVector<FT, OAF, OPF, !SO> &rhs)\
{\
    return norm##Norm(rhs - trans(lhs));\
}\
\
template<typename MatType1, typename MatType2, bool SO, bool isDense, bool isDense2, bool isSymmetric>\
INLINE double norm##Dist(const blaze::Row<MatType1, SO, isDense, isSymmetric> &lhs,\
                      const blaze::Row<MatType2, SO, isDense2, isSymmetric> &rhs)\
{\
    return norm##Norm(rhs - lhs);\
}\
\
template<typename MatType1, typename MatType2, bool SO, bool isDense, bool isDense2, bool isSymmetric>\
INLINE double norm##Dist(const blaze::Column<MatType1, SO, isDense, isSymmetric> &lhs,\
                      const blaze::Column<MatType2, SO, isDense2, isSymmetric> &rhs)\
{\
    return norm##Norm(rhs - lhs);\
}
DECL_DIST(l1)
DECL_DIST(l2)
DECL_DIST(sqr)
DECL_DIST(l3)
DECL_DIST(l4)
DECL_DIST(max)
DECL_DIST(inf)
#undef DECL_DIST
template<typename FT, typename A, typename OA>
inline double l2Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return l2Dist(CustomVector<const FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<const FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}

template<typename FT, bool SO>
inline double sqrL2Dist(const blz::DynamicVector<FT, SO> &v1, const blz::DynamicVector<FT, SO> &v2) {
    return l2Dist(v1, v2);
}
template<typename FT, bool AF, bool PF, bool SO, bool OAF, bool OPF, bool OSO>
inline double sqrL2Dist(const blz::CustomVector<FT, AF, PF, SO> &v1, const blz::CustomVector<FT, OAF, OPF, OSO> &v2) {
    return l2Dist(v1, v2);
}

template<typename FT, typename A, typename OA>
inline double sqrL2Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return sqrL2Dist(CustomVector<const FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                     CustomVector<const FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}

template<typename FT, typename A, typename OA>
INLINE double sqrDist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return sqrL2Dist(lhs, rhs);
}

template<typename FT, typename A, typename OA>
inline double l1Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return l1Dist(CustomVector<const FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<const FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}
template<typename FT, typename A, typename OA>
inline double l3Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return l3Dist(CustomVector<const FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<const FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}

template<typename FT, typename A, typename OA>
inline double l4Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return l4Dist(CustomVector<FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}
template<typename FT, typename A, typename OA>
inline double maxDist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return maxDist(CustomVector<FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}
template<typename FT, typename A, typename OA>
inline double infDist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {return maxDist(lhs, rhs);}

template<typename Base>
struct sqrBaseNorm: public Base {
    template<typename C1, typename C2>
    INLINE constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        return std::pow(Base::operator()(lhs, rhs), 2);
    }
};
struct L1Norm {
    template<typename C1, typename C2>
    INLINE constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        return l1Dist(lhs, rhs);
    }
};
struct L2Norm {
    template<typename C1, typename C2>
    INLINE constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        return l2Dist(lhs, rhs);
    }
};
struct sqrL2Norm {
    template<typename C1, typename C2>
    INLINE constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        return sqrDist(lhs, rhs);
    }
};
struct L3Norm {
    template<typename C1, typename C2>
    INLINE constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        return l3Dist(lhs, rhs);
    }
};
struct L4Norm {
    template<typename C1, typename C2>
    INLINE constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        return l4Dist(lhs, rhs);
    }
};
struct maxNormFunctor {
    template<typename C1, typename C2>
    INLINE constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        return maxDist(lhs, rhs);
    }
};
struct infNormFunction: maxNormFunctor{};
struct sqrL1Norm: sqrBaseNorm<L1Norm> {};
struct sqrL3Norm: sqrBaseNorm<L3Norm> {};
struct sqrL4Norm: sqrBaseNorm<L4Norm> {};
struct sqrMaxNorm: sqrBaseNorm<maxNormFunctor> {};

/*
 *
 * Use https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf
 * to derive KL, Jensen-Shannon divergences + JS distances (sqrt(JS))
 * for the full exponential family of distributions.
 * Currently, what I have is the multinomial, but this also applies to Poisson.
 * See https://en.wikipedia.org/wiki/Exponential_family
 * TODO:
 * These include:
 * Bernoulli, binomial, Poisson, chi-squared, Laplace, normal, lognormal, inverse gaussian, gamma,
 * inverse gamma, beta, categorical, Dirichlet, Wishart
 */

template<typename FT, bool SO>
double logsumexp(const blaze::DenseVector<FT, SO> &x) {
    auto maxv = blaze::max(~x);
    return maxv + std::log(blaze::sum(blaze::exp(~x - maxv)));
}
template<typename FT, bool SO>
double logsumexp(const blaze::SparseVector<FT, SO> &x) {
    auto maxv = blaze::max(~x);
    double s = 0.;
    blaze::CompressedVector<typename FT::ElementType, SO> cv(~x);
    for(const auto p: cv) {
        //auto p = *it++;
        s += std::exp(p.value() - maxv); // Sum over sparse elements
    }
    s += nonZeros(~x) * std::exp(-maxv);  // Handle the ones we skipped
    return maxv + std::log(s);
}

template<typename VT>
INLINE double multinomial_cumulant(const VT &x) {return logsumexp(x);}

template<typename FT, bool SO, typename OFT>
double multinomial_jsd(const blaze::DenseVector<FT, SO> &lhs, const blaze::DenseVector<FT, SO> &rhs, OFT lhc, OFT rhc) {
    // Note: multinomial cumulants for lhs and rhs can be cached as lhc/rhc, such that
    // multinomial_cumulant(mean) and the dot products are all that are required
    // TODO: optimize for sparse vectors (maybe filt can be eliminated?)
    // TODO: cache logs as well as full vectors?

    // We can turn inf values into 0 because:
    // Whenever P ( x ) {P(x)} P(x) is zero the contribution of the corresponding term is interpreted as zero because
    // lim_{x->0+}[xlog(x)] = 0
    // See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Definition
    using FloatType = typename FT::ElementType;

    auto mean = (~lhs + ~rhs) * .5;
    auto fi = [](auto x) {return std::isinf(x) ? FloatType(0): x;};
    auto logmean = blaze::map(blaze::log(mean), fi);
    auto lhterm = blaze::map(blaze::log(~lhs) - logmean, fi);
    auto rhterm = blaze::map(blaze::log(~rhs) - logmean, fi);
    auto lhv = dot(lhterm, ~lhs), rhv = dot(rhterm, ~rhs);
    const auto retsq = multinomial_cumulant(mean) + (lhv + rhv - lhc - rhc) * .5;
    return std::sqrt(retsq);
}

template<typename FT, bool SO, typename OFT>
double multinomial_jsd(const blaze::DenseVector<FT, SO> &lhs, const blaze::DenseVector<FT, SO> &rhs,
                       // Original vectors
                       const blaze::DenseVector<FT, SO> &lhl, const blaze::DenseVector<FT, SO> &rhl,
                       // cached logs
                       OFT lhc, OFT rhc) {
    using FloatType = typename FT::ElementType;
    assert(std::find_if(lhl.begin(), lhl.end(), [](auto x)
                        {return std::isinf(x) || std::isnan(x);})
           == lhl.end());
    assert(std::find_if(rhl.begin(), rhl.end(), [](auto x)
                        {return std::isinf(x) || std::isnan(x);})
           == rhl.end());
    auto mean = (~lhs + ~rhs) * .5;
    auto fi = [](auto x) {return std::isinf(x) ? FloatType(0): x;};
    auto logmean = blaze::map(blaze::log(mean), fi);
    return std::sqrt(
               multinomial_cumulant(mean)
               + .5 * (
                   + dot(blaze::map(lhl - logmean, fi), ~lhs)
                   + dot(blaze::map(rhl - logmean, fi), ~rhs)
                   - lhc - rhc
               )
           );
}

template<typename FT, bool SO, typename LT, typename OFT>
double multinomial_jsd(const blaze::SparseVector<FT, SO> &lhs, const blaze::SparseVector<FT, SO> &rhs,
                       // Original vectors
                       const LT &lhl, const LT &rhl,
                       //const blaze::SparseVector<FT, SO> &lhl, const blaze::SparseVector<FT, SO> &rhl,
                       // cached logs
                       OFT lhc, OFT rhc) {
    assert(std::find_if(lhl.begin(), lhl.end(), [](auto x)
                        {return std::isinf(x.value()) || std::isnan(x.value());})
           == lhl.end());
    assert(std::find_if(rhl.begin(), rhl.end(), [](auto x)
                        {return std::isinf(x.value()) || std::isnan(x.value());})
           == rhl.end());
    auto mean = (~lhs + ~rhs) * .5;
    auto logmean = blaze::log(mean);
    return std::sqrt(
               multinomial_cumulant(mean)
               + .5 * (
                   + dot(lhl - logmean, ~lhs)
                   + dot(rhl - logmean, ~rhs)
                   - lhc - rhc
               )
           );
}

template<typename FT, bool SO>
double multinomial_jsd(const blaze::SparseVector<FT, SO> &lhs, const blaze::SparseVector<FT, SO> &rhs) {
    auto lhl = blaze::log(lhs);
    auto rhl = blaze::log(rhs);
    return multinomial_jsd(lhs, rhs, lhl, rhl, multinomial_cumulant(lhs), multinomial_cumulant(rhs));
}
    

template<typename FT, bool SO>
INLINE double multinomial_jsd(const blaze::DenseVector<FT, SO> &lhs,
                              const blaze::DenseVector<FT, SO> &rhs)
{
    assert((~lhs).size() == (~rhs).size());
    return multinomial_jsd(lhs, rhs, multinomial_cumulant(~lhs), multinomial_cumulant(~rhs));
}


template<typename LHVec, typename RHVec>
double bhattacharya_measure(const LHVec &lhs, const RHVec &rhs) {
    // Requires same storage.
    // TODO: generalize for different storage classes/transpose flags using DenseVector and SparseVector
    // base classes
    return sqrt(lhs * rhs);
}

template<typename LHVec, typename RHVec>
double bhattacharya_metric(const LHVec &lhs, const RHVec &rhs) {
    // Comaniciu, D., Ramesh, V. & Meer, P. (2003). Kernel-based object tracking.IEEE Transactionson Pattern Analysis and Machine Intelligence,25(5), 564-577.
    // Proves that this extension is a valid metric
    // See http://www.cse.yorku.ca/~kosta/CompVis_Notes/bhattacharyya.pdf
    return std::sqrt(1. - bhattacharya_measure(lhs, rhs));
}
template<typename LHVec, typename RHVec>
double matusita_distance(const LHVec &lhs, const RHVec &rhs) {
    return sqrL2Dist(sqrt(lhs), sqrt(rhs));
}
} // distance

} // namespace blz

#endif // FGC_DISTANCE_AND_MEANING_H__
