#ifndef FGC_DISTANCE_AND_MEANING_H__
#define FGC_DISTANCE_AND_MEANING_H__
#include "blaze_adaptor.h"
#include "boost/iterator/transform_iterator.hpp"
#include <vector>
#include <iostream>

namespace blz {

inline namespace distance {
#define DECL_DIST(norm) \
template<typename FT, bool SO>\
INLINE double norm##Dist(const blaze::DynamicVector<FT, SO> &lhs, const blaze::DynamicVector<FT, SO> &rhs) {\
    return norm##Norm(rhs - lhs);\
}\
template<typename VT, typename VT2, bool SO>\
INLINE double norm##Dist(const blaze::DenseVector<VT, SO> &lhs, const blaze::DenseVector<VT2, SO> &rhs) {\
    return norm##Norm(~rhs - ~lhs);\
}\
\
template<typename VT, typename VT2, bool SO>\
INLINE double norm##Dist(const blaze::DenseVector<VT, SO> &lhs, const blaze::DenseVector<VT2, !SO> &rhs) {\
    return norm##Norm(~rhs - trans(~lhs));\
}\
\
template<typename VT, typename VT2, bool SO>\
INLINE double norm##Dist(const blaze::SparseVector<VT, SO> &lhs, const blaze::SparseVector<VT2, SO> &rhs) {\
    return norm##Norm(~rhs - ~lhs);\
}\
\
template<typename VT, typename VT2, bool SO>\
INLINE double norm##Dist(const blaze::SparseVector<VT, SO> &lhs, const blaze::SparseVector<VT2, !SO> &rhs) {\
    return norm##Norm(~rhs - trans(~lhs));\
}\


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

template<typename FT, typename FT2, bool SO>
inline double sqrL2Dist(const blz::Vector<FT, SO> &v1, const blz::Vector<FT2, SO> &v2) {
    return l2Dist(~v1, ~v2);
}
template<typename FT, blaze::AlignmentFlag AF, blaze::PaddingFlag PF, bool SO, blaze::AlignmentFlag OAF, blaze::PaddingFlag OPF, bool OSO>
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


// For D^2 sampling.
template<typename BaseDist>
struct SqrNormFunctor: public BaseDist {
    template<typename...Args> SqrNormFunctor(Args &&...args): BaseDist(std::forward<Args>(args)...) {}
    template<typename C1, typename C2>
    INLINE constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        double basedist = BaseDist::operator()(lhs, rhs);
        return basedist * basedist;
    }
};
template<>
struct SqrNormFunctor<L2Norm>: public sqrL2Norm {};

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
 */

template<typename FT, bool SO>
double logsumexp(const blaze::DenseVector<FT, SO> &x) {
    const auto maxv = blaze::max(~x);
    return maxv + std::log(blaze::sum(blaze::exp(~x - maxv)));
}

template<typename FT, bool SO>
double logsumexp(const blaze::SparseVector<FT, SO> &x) {
    auto maxv = blaze::max(~x);
    double s = 0.;
    for(const auto p: ~x) {
        s += std::exp(p.value() - maxv); // Sum over sparse elements
    }
    s += ((~x).size() - nonZeros(~x)) * std::exp(-maxv);  // Handle the ones we skipped
    return maxv + std::log(s);
}

template<typename VT1, typename VT2, typename Scalar, bool TF>
double logsumexp(const SVecScalarMultExpr<SVecSVecAddExpr<VT1, VT2, TF>, Scalar, TF> &exp) {
    // Specifically for calculating the logsumexp of the mean of the two sparse vectors.
    // SVecScalarMultExpr doesn't provide a ConstIterator, so we're rolling our own specialized function.
    auto maxv = blaze::max(~exp), mmax = -maxv;
    double s = 0.;
    auto lit = exp.leftOperand().leftOperand().begin(), rit = exp.leftOperand().rightOperand().begin(),
         lie = exp.leftOperand().leftOperand().end(),   rie = exp.leftOperand().rightOperand().end();
    Scalar scalar = exp.rightOperand();
    auto f = [scalar,mmax,&s](auto x) {s += std::exp(std::fma(x, scalar, mmax));};
    size_t nnz = 0;
    for(;;) {
        if(lit == lie) {
            while(rit != rie) {
                f((rit++)->value());
                ++nnz;
            }
            break;
        }
        if(rit == rie) {
            while(lit != lie) {
                f((lit++)->value());
                ++nnz;
            }
            break;
        }
        if(lit->index() < rit->index()) {
            f((lit++)->value());
            ++nnz;
        } else if(rit->index() < lit->index()) {
            f((rit++)->value());
            ++nnz;
        } else /* indexes match */ {
            ++nnz;
            f(lit->value() + rit->value());
            ++lit, ++rit;
        }
    }
    s += ((~exp).size() - nnz) * std::exp(-maxv);  // Handle the ones we skipped
    return maxv + std::log(s);
}

    /* Note: multinomial cumulants for lhs and rhs can be cached as lhc/rhc, such that
    *  multinomial_cumulant(mean) and the dot products are all that are required
    *  TODO: optimize for sparse vectors (maybe filt can be eliminated?)
    *  TODO: cache logs as well as full vectors?

    *  We can turn inf values into 0 because:
    *  Whenever P ( x ) {P(x)} P(x) is zero the contribution of the corresponding term is interpreted as zero because
    *  lim_{x->0+}[xlog(x)] = 0
    *  See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Definition

    *  Alternatively, we can use the Dirichlet prior and avoid any 0s in the first place.
    *  This is done by replacing parameters mu_n = \frac{n}{\sum_{n \in N}{n}})
    *  with                                 mu_n = \frac{n+1}{\sum_{n \in N}{n + 1}}
    *  If this has been done, set filter_nans to be false in any of the downstream functions
    *  See https://mathoverflow.net/questions/72668/how-to-compute-kl-divergence-when-pmf-contains-0s
    *  Another possible prior is the Gamma prior with both parameters $\Beta, \Beta$,
    *  which replaces mu_n with $\frac{n+\Beta}{\sum_{n \in N}{n + \Beta}}$,
    *  as in https://arxiv.org/abs/1202.6201.
    */

template<typename VT>
INLINE double multinomial_cumulant(const VT &x) {return logsumexp(x);}

template<bool VT>
struct FilterNans {
    static constexpr bool value = VT;
};

namespace bnj {

template<typename VT, typename VT2, bool SO>
INLINE double multinomial_jsd(const blaze::DenseVector<VT, SO> &lhs,
                              const blaze::DenseVector<VT, SO> &rhs,
                              const blaze::DenseVector<VT2, SO> &lhlog,
                              const blaze::DenseVector<VT2, SO> &rhlog)
{
#if 0
    assert_all_nonzero(lhs);
    assert_all_nonzero(rhs);
#endif
    auto mn = (~lhs + ~rhs) * 0.5;
    auto mnlog = blaze::evaluate(blaze::neginf2zero(blaze::log(~mn)));
    double lhc = blaze::dot(~lhs, ~lhlog - mnlog);
    double rhc = blaze::dot(~rhs, ~rhlog - mnlog);
    return 0.5 * (lhc + rhc);
}

template<typename VT, typename VT2, bool SO>
INLINE double multinomial_jsd(const blaze::SparseVector<VT, SO> &lhs,
                              const blaze::SparseVector<VT, SO> &rhs,
                              const blaze::SparseVector<VT2, SO> &lhlog,
                              const blaze::SparseVector<VT2, SO> &rhlog)
{
    auto mnlog = blaze::evaluate(blaze::log((~lhs + ~rhs) * 0.5));
    double lhc = blaze::dot(~lhs, ~lhlog - mnlog);
    double rhc = blaze::dot(~rhs, ~rhlog - mnlog);
    return 0.5 * (lhc + rhc);
}
template<typename VT, bool SO>
INLINE double multinomial_jsd(const blaze::DenseVector<VT, SO> &lhs,
                              const blaze::DenseVector<VT, SO> &rhs)
{
    auto lhlog = blaze::evaluate(neginf2zero(log(lhs)));
    auto rhlog = blaze::evaluate(neginf2zero(log(rhs)));
    auto mnlog = blaze::evaluate(neginf2zero(log((~lhs + ~rhs) * 0.5)));
    double lhc = blaze::dot(~lhs, ~lhlog - mnlog);
    double rhc = blaze::dot(~rhs, ~rhlog - mnlog);
    return 0.5 * (lhc + rhc);
}
template<typename VT, typename VT2, bool SO>
INLINE double multinomial_jsd(const blaze::Vector<VT, SO> &lhs,
                              const blaze::Vector<VT2, SO> &rhs)
{
    auto lhlog = blaze::evaluate(neginf2zero(log(~lhs)));
    auto rhlog = blaze::evaluate(neginf2zero(log(~rhs)));
    auto mnlog = blaze::evaluate(neginf2zero(log((~lhs + ~rhs) * 0.5)));
    double lhc = blaze::dot(~lhs, ~lhlog - mnlog);
    double rhc = blaze::dot(~rhs, ~rhlog - mnlog);
    return 0.5 * (lhc + rhc);
}
template<typename VT, bool SO>
INLINE double multinomial_jsd(const blaze::SparseVector<VT, SO> &lhs,
                              const blaze::SparseVector<VT, SO> &rhs)
{
    auto lhlog = blaze::log(lhs), rhlog = blaze::log(rhs);
    auto mnlog = blaze::evaluate(blaze::log((~lhs + ~rhs) * 0.5));
    double lhc = blaze::dot(~lhs, ~lhlog - mnlog);
    double rhc = blaze::dot(~rhs, ~rhlog - mnlog);
    return 0.5 * (lhc + rhc);
}

template<typename FT, bool SO>
INLINE double multinomial_bregman(const blaze::DenseVector<FT, SO> &lhs,
                                  const blaze::DenseVector<FT, SO> &rhs,
                                  const blaze::DenseVector<FT, SO> &lhlog,
                                  const blaze::DenseVector<FT, SO> &rhlog)
{
    assert_all_nonzero(lhs);
    assert_all_nonzero(rhs);
    return blaze::dot(lhs, lhlog - rhlog);
}
template<typename FT, bool SO>
INLINE double      poisson_bregman(const blaze::DenseVector<FT, SO> &lhs,
                                   const blaze::DenseVector<FT, SO> &rhs,
                                   const blaze::DenseVector<FT, SO> &lhlog,
                                   const blaze::DenseVector<FT, SO> &rhlog)
{
    // Assuming these are all independent (not ideal)
    assert_all_nonzero(lhs);
    assert_all_nonzero(rhs);
    return blaze::dot(lhs, lhlog - rhlog) + blaze::sum(rhs - lhs);
}
} // namespace bnj
using namespace bnj;

enum Prior {
    NONE,       // Do not modify values. Requires that there are no nonzero parameters
    DIRICHLET,  // Uniform smoothing
    GAMMA_BETA, // Requires a parameter
    FEATURE_SPECIFIC_PRIOR // Requires a vector of parameters
};



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
template<typename...Args>
INLINE decltype(auto) multinomial_jsm(Args &&...args) {
    using blaze::sqrt;
    using std::sqrt;
    return sqrt(multinomial_jsd(std::forward<Args>(args)...));
}

template<typename VT, bool SO, typename VT2>
auto p_wasserstein(const blz::DenseVector<VT, SO> &x, const blz::DenseVector<VT2, SO> &y, double p=1.) {
    auto &xr = ~x;
    auto &yr = ~y;
	const size_t sz = xr.size();
	std::unique_ptr<uint32_t[]> ptr(new uint32_t[sz * 2]);
	auto xptr = ptr.get(), yptr = ptr.get() + xr.size();
	std::iota(xptr, yptr, 0u);
	std::iota(yptr, yptr + sz, 0u);
	pdqsort(xptr, xptr + sz, [xdat=xr.data()](uint32_t p, uint32_t q) {return xdat[p] < xdat[q];});
	pdqsort(yptr, yptr + sz, [ydat=yr.data()](uint32_t p, uint32_t q) {return ydat[p] < ydat[q];});
	auto xconv = [xdat=xr.data()](auto x) {return xdat[x];};
	auto yconv = [ydat=yr.data()](auto y) {return ydat[y];};
    blz::DynamicVector<typename VT::ElementType, SO> all(2 * sz);
	std::merge(boost::make_transform_iterator(xptr, xconv), boost::make_transform_iterator(xptr + sz, xconv),
	           boost::make_transform_iterator(yptr, yconv), boost::make_transform_iterator(yptr + sz, yconv), all.begin());
    const size_t deltasz = 2 * sz - 1;
    blz::DynamicVector<typename VT::ElementType, SO> deltas(deltasz);
    std::adjacent_difference(all.begin(), all.end(), deltas.begin());
    blz::DynamicVector<typename VT::ElementType> cdfx(deltasz), cdfy(deltasz);
    size_t xoffset = 0;
    for(size_t i = 0; i < cdfx.size(); ++i) {
        while(xoffset < sz && xptr[xoffset] > all[i])
            ++xoffset;
        if(xoffset == sz) {
            std::fill_n(cdfx.data() + i, deltasz - i, sz);
            break;
        }
        cdfx[i] = xoffset;
    }
    xoffset = 0;
    for(size_t i = 0; i < cdfy.size(); ++i) {
        while(xoffset < sz && yptr[xoffset] > all[i])
            ++xoffset;
        if(xoffset == sz) {
            std::fill_n(cdfy.data() + i, deltasz - i, sz);
            break;
        }
        cdfy[i] = xoffset;
    }
	if(p == 1.)
		return dot(blz::abs(cdfx - cdfy), deltas) / sz;
    cdfx *= 1. / sz;
    cdfy *= 1. / sz;
    if(p == 2)
		return std::sqrt(dot(blz::pow(cdfx - cdfy, 2), deltas));
	return std::pow(dot(blz::pow(blz::abs(cdfx - cdfy), p), deltas), 1. / p);
}

} // distance

} // namespace blz

#endif // FGC_DISTANCE_AND_MEANING_H__
