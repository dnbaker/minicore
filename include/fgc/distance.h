#ifndef FGC_DISTANCE_AND_MEANING_H__
#define FGC_DISTANCE_AND_MEANING_H__
#include <vector>
#include <iostream>
#include <set>


#include "blaze_adaptor.h"

#ifndef BOOST_NO_AUTO_PTR
#define BOOST_NO_AUTO_PTR 1
#endif

#include "network_simplex/network_simplex_simple.h"
#include "boost/iterator/transform_iterator.hpp"

namespace blz {

inline namespace distance {

enum ProbDivType {
    L1,
    L2,
    SQRL2,
    JSM, // Multinomial Jensen-Shannon Metric
    JSD, // Multinomial Jensen-Shannon Divergence
    MKL, // Multinomial KL Divergence
    POISSON, // Poisson KL
    HELLINGER,
    BHATTACHARYYA_METRIC,
    BHATTACHARYYA_DISTANCE,
    TOTAL_VARIATION_DISTANCE,
    LLR,
    EMD,
    WEMD, // Weighted Earth-mover's distance
    REVERSE_MKL,
    REVERSE_POISSON,
    UWLLR, /* Unweighted Log-likelihood Ratio.
            * Specifically, this is the D_{JSD}^{\lambda}(x, y),
            * where \lambda = \frac{N_p}{N_p + N_q}
            *
            */
    OLLR,       // Old LLR, deprecated (included for compatibility/comparisons)
    ITAKURA_SAITO, // \sum_{i=1}^D[\frac{a_i}{b_i} - \log{\frac{a_i}{b_i}} - 1]
    REVERSE_ITAKURA_SAITO, // Reverse I-S
    WLLR = LLR, // Weighted Log-likelihood Ratio, now equivalent to the LLR
    TVD = TOTAL_VARIATION_DISTANCE,
    WASSERSTEIN=EMD,
    PSD = JSD, // Poisson JSD, but algebraically equivalent
    PSM = JSM,
    IS=ITAKURA_SAITO
};
namespace detail {
static constexpr INLINE bool  needs_logs(ProbDivType d)  {
    switch(d) {
        case JSM: case JSD: case MKL: case POISSON: case LLR: case OLLR: case ITAKURA_SAITO:
        case REVERSE_MKL: case REVERSE_POISSON: case UWLLR: case REVERSE_ITAKURA_SAITO: return true;
        default: break;
    }
    return false;
}


static constexpr INLINE bool  needs_sqrt(ProbDivType d) {
    return d == HELLINGER || d == BHATTACHARYYA_METRIC || d == BHATTACHARYYA_DISTANCE;
}

static constexpr INLINE bool is_symmetric(ProbDivType d) {
    switch(d) {
        case L1: case L2: case EMD: case HELLINGER: case BHATTACHARYYA_DISTANCE: case BHATTACHARYYA_METRIC:
        case JSD: case JSM: case LLR: case UWLLR: case SQRL2: case TOTAL_VARIATION_DISTANCE: case OLLR:
            return true;
        default: ;
    }
    return false;
}

static constexpr INLINE const char *prob2str(ProbDivType d) {
    switch(d) {
        case BHATTACHARYYA_DISTANCE: return "BHATTACHARYYA_DISTANCE";
        case BHATTACHARYYA_METRIC: return "BHATTACHARYYA_METRIC";
        case EMD: return "EMD";
        case HELLINGER: return "HELLINGER";
        case JSD: return "JSD/PSD";
        case JSM: return "JSM/PSM";
        case L1: return "L1";
        case L2: return "L2";
        case LLR: return "LLR";
        case OLLR: return "OLLR";
        case UWLLR: return "UWLLR";
        case ITAKURA_SAITO: return "ITAKURA_SAITO";
        case MKL: return "MKL";
        case POISSON: return "POISSON";
        case REVERSE_MKL: return "REVERSE_MKL";
        case REVERSE_POISSON: return "REVERSE_POISSON";
        case REVERSE_ITAKURA_SAITO: return "REVERSE_ITAKURA_SAITO";
        case SQRL2: return "SQRL2";
        case TOTAL_VARIATION_DISTANCE: return "TOTAL_VARIATION_DISTANCE";
        default: return "INVALID TYPE";
    }
}
static constexpr INLINE const char *prob2desc(ProbDivType d) {
    switch(d) {
        case BHATTACHARYYA_DISTANCE: return "Bhattacharyya distance: -log(dot(sqrt(x) * sqrt(y)))";
        case BHATTACHARYYA_METRIC: return "Bhattacharyya metric: sqrt(1 - BhattacharyyaSimilarity(x, y))";
        case EMD: return "Earth Mover's Distance: Optimal Transport";
        case HELLINGER: return "Hellinger Distance: sqrt(sum((sqrt(x) - sqrt(y))^2))/2";
        case JSD: return "Jensen-Shannon Divergence for Poisson and Multinomial models, for which they are equivalent";
        case JSM: return "Jensen-Shannon Metric, known as S2JSD and the Endres metric, for Poisson and Multinomial models, for which they are equivalent";
        case L1: return "L1 distance";
        case L2: return "L2 distance";
        case LLR: return "Log-likelihood Ratio under the multinomial model";
        case OLLR: return "Original log-likelihood ratio. This is likely not correct, but it is related to the Jensen-Shannon Divergence";
        case UWLLR: return "Unweighted Log-likelihood Ratio. This is effectively the Generalized Jensen-Shannon Divergence with lambda parameter corresponding to the fractional contribution of counts in the first observation. This is symmetric, unlike the G_JSD, because the parameter comes from the counts.";
        case MKL: return "Multinomial KL divergence";
        case POISSON: return "Poisson KL Divergence";
        case REVERSE_MKL: return "Reverse Multinomial KL divergence";
        case REVERSE_POISSON: return "Reverse KL divergence";
        case SQRL2: return "Squared L2 Norm";
        case TOTAL_VARIATION_DISTANCE: return "Total Variation Distance: 1/2 sum_{i in D}(|x_i - y_i|)";
        case ITAKURA_SAITO: return "Itakura-Saito divergence, a Bregman divergence [sum((a / b) - log(a / b) - 1 for a, b in zip(A, B))]";
        case REVERSE_ITAKURA_SAITO: return "Reversed Itakura-Saito divergence, a Bregman divergence";
        default: return "INVALID TYPE";
    }
}
static void print_measures() {
    std::set<ProbDivType> measures {
        L1,
        L2,
        SQRL2,
        JSM,
        JSD,
        MKL,
        POISSON,
        HELLINGER,
        BHATTACHARYYA_METRIC,
        BHATTACHARYYA_DISTANCE,
        TOTAL_VARIATION_DISTANCE,
        LLR,
        OLLR,
        EMD,
        REVERSE_MKL,
        REVERSE_POISSON,
        UWLLR,
        TOTAL_VARIATION_DISTANCE,
        WASSERSTEIN,
        PSD,
        PSM,
        ITAKURA_SAITO
    };
    for(const auto measure: measures) {
        std::fprintf(stderr, "Code: %d. Description: '%s'. Short name: '%s'\n", measure, prob2desc(measure), prob2str(measure));
    }
}
} // detail


#define DECL_DIST(norm) \
template<typename FT, bool SO>\
INLINE auto norm##Dist(const blaze::DynamicVector<FT, SO> &lhs, const blaze::DynamicVector<FT, SO> &rhs) {\
    return norm##Norm(rhs - lhs);\
}\
template<typename VT, typename VT2, bool SO>\
INLINE auto norm##Dist(const blaze::DenseVector<VT, SO> &lhs, const blaze::DenseVector<VT2, SO> &rhs) {\
    return norm##Norm(~rhs - ~lhs);\
}\
\
template<typename VT, typename VT2, bool SO>\
INLINE auto norm##Dist(const blaze::DenseVector<VT, SO> &lhs, const blaze::DenseVector<VT2, !SO> &rhs) {\
    return norm##Norm(~rhs - trans(~lhs));\
}\
\
template<typename VT, typename VT2, bool SO>\
INLINE auto norm##Dist(const blaze::SparseVector<VT, SO> &lhs, const blaze::SparseVector<VT2, SO> &rhs) {\
    return norm##Norm(~rhs - ~lhs);\
}\
\
template<typename VT, typename VT2, bool SO>\
INLINE auto norm##Dist(const blaze::SparseVector<VT, SO> &lhs, const blaze::SparseVector<VT2, !SO> &rhs) {\
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
inline auto l2Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return l2Dist(CustomVector<const FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<const FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}

template<typename FT, typename FT2, bool SO>
inline auto sqrL2Dist(const blz::Vector<FT, SO> &v1, const blz::Vector<FT2, SO> &v2) {
    return sqrDist(~v1, ~v2);
}
template<typename FT, blaze::AlignmentFlag AF, blaze::PaddingFlag PF, bool SO, blaze::AlignmentFlag OAF, blaze::PaddingFlag OPF, bool OSO>
inline auto sqrL2Dist(const blz::CustomVector<FT, AF, PF, SO> &v1, const blz::CustomVector<FT, OAF, OPF, OSO> &v2) {
    return sqrDist(v1, v2);
}

template<typename FT, typename A, typename OA>
inline auto sqrL2Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return sqrL2Dist(CustomVector<const FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                     CustomVector<const FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}

template<typename FT, typename A, typename OA>
INLINE auto sqrDist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return sqrL2Dist(lhs, rhs);
}

template<typename FT, typename A, typename OA>
inline auto l1Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return l1Dist(CustomVector<const FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<const FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}
template<typename FT, typename A, typename OA>
inline auto l3Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return l3Dist(CustomVector<const FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<const FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}

template<typename FT, typename A, typename OA>
inline auto l4Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return l4Dist(CustomVector<FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}
template<typename FT, typename A, typename OA>
inline auto maxDist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return maxDist(CustomVector<FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}
template<typename FT, typename A, typename OA>
inline auto infDist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {return maxDist(lhs, rhs);}

template<typename Base>
struct sqrBaseNorm: public Base {
    template<typename C1, typename C2>
    INLINE constexpr auto operator()(const C1 &lhs, const C2 &rhs) const {
        return std::pow(Base::operator()(lhs, rhs), 2);
    }
};
struct L1Norm {
    template<typename C1, typename C2>
    INLINE constexpr auto operator()(const C1 &lhs, const C2 &rhs) const {
        return l1Dist(lhs, rhs);
    }
};
struct L2Norm {
    template<typename C1, typename C2>
    INLINE constexpr auto operator()(const C1 &lhs, const C2 &rhs) const {
        return l2Dist(lhs, rhs);
    }
};
struct sqrL2Norm {
    template<typename C1, typename C2>
    INLINE constexpr auto operator()(const C1 &lhs, const C2 &rhs) const {
        return sqrDist(lhs, rhs);
    }
};
struct L3Norm {
    template<typename C1, typename C2>
    INLINE constexpr auto operator()(const C1 &lhs, const C2 &rhs) const {
        return l3Dist(lhs, rhs);
    }
};
struct L4Norm {
    template<typename C1, typename C2>
    INLINE constexpr auto operator()(const C1 &lhs, const C2 &rhs) const {
        return l4Dist(lhs, rhs);
    }
};
struct maxNormFunctor {
    template<typename C1, typename C2>
    INLINE constexpr auto operator()(const C1 &lhs, const C2 &rhs) const {
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
    INLINE constexpr auto operator()(const C1 &lhs, const C2 &rhs) const {
        auto basedist = BaseDist::operator()(lhs, rhs);
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
auto logsumexp(const blaze::DenseVector<FT, SO> &x) {
    const auto maxv = blaze::max(~x);
    return maxv + std::log(blaze::sum(blaze::exp(~x - maxv)));
}

template<typename FT, bool SO>
auto logsumexp(const blaze::SparseVector<FT, SO> &x) {
    auto maxv = blaze::max(~x);
    auto s = 0.;
    for(const auto p: ~x) {
        s += std::exp(p.value() - maxv); // Sum over sparse elements
    }
    s += ((~x).size() - nonZeros(~x)) * std::exp(-maxv);  // Handle the ones we skipped
    return maxv + std::log(s);
}

template<typename VT1, typename VT2, typename Scalar, bool TF>
auto logsumexp(const SVecScalarMultExpr<SVecSVecAddExpr<VT1, VT2, TF>, Scalar, TF> &exp) {
    // Specifically for calculating the logsumexp of the mean of the two sparse vectors.
    // SVecScalarMultExpr doesn't provide a ConstIterator, so we're rolling our own specialized function.
    auto maxv = blaze::max(~exp), mmax = -maxv;
    auto s = 0.;
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
INLINE auto multinomial_cumulant(const VT &x) {return logsumexp(x);}

template<bool VT>
struct FilterNans {
    static constexpr bool value = VT;
};

namespace bnj {

template<typename VT, typename VT2, bool SO, typename RT=blz::CommonType_t<blz::ElementType_t<VT>, blz::ElementType_t<VT2>> >
INLINE auto multinomial_jsd(const blaze::DenseVector<VT, SO> &lhs,
                            const blaze::DenseVector<VT, SO> &rhs,
                            const blaze::DenseVector<VT2, SO> &lhlog,
                            const blaze::DenseVector<VT2, SO> &rhlog)
{
    auto mn = (~lhs + ~rhs) * RT(0.5);
    auto mnlog = blaze::evaluate(blaze::neginf2zero(blaze::log(~mn)));
    auto lhc = blaze::dot(~lhs, ~lhlog - mnlog);
    auto rhc = blaze::dot(~rhs, ~rhlog - mnlog);
    return RT(0.5) * (lhc + rhc);
}

template<typename VT, typename VT2, bool SO>
INLINE auto multinomial_jsd(const blaze::SparseVector<VT, SO> &lhs,
                            const blaze::SparseVector<VT, SO> &rhs,
                            const blaze::SparseVector<VT2, SO> &lhlog,
                            const blaze::SparseVector<VT2, SO> &rhlog)
{
    using RT = blz::CommonType_t<blz::ElementType_t<VT>, blz::ElementType_t<VT2>>;
    auto mnlog = blaze::evaluate(blaze::log((~lhs + ~rhs) * RT(0.5)));
    auto lhc = blaze::dot(~lhs, ~lhlog - mnlog);
    auto rhc = blaze::dot(~rhs, ~rhlog - mnlog);
    return RT(0.5) * (lhc + rhc);
}
template<typename VT, bool SO>
INLINE auto multinomial_jsd(const blaze::DenseVector<VT, SO> &lhs,
                            const blaze::DenseVector<VT, SO> &rhs)
{
    using RT = blz::ElementType_t<VT>;
    auto lhlog = blaze::evaluate(neginf2zero(log(lhs)));
    auto rhlog = blaze::evaluate(neginf2zero(log(rhs)));
    auto mnlog = blaze::evaluate(neginf2zero(log((~lhs + ~rhs) * RT(0.5))));
    auto lhc = blaze::dot(~lhs, ~lhlog - mnlog);
    auto rhc = blaze::dot(~rhs, ~rhlog - mnlog);
    return RT(0.5) * (lhc + rhc);
}
template<typename VT, typename VT2, bool SO>
INLINE auto multinomial_jsd(const blaze::Vector<VT, SO> &lhs,
                            const blaze::Vector<VT2, SO> &rhs)
{
    using RT = blz::ElementType_t<VT>;
    auto lhlog = blaze::evaluate(neginf2zero(log(~lhs)));
    auto rhlog = blaze::evaluate(neginf2zero(log(~rhs)));
    auto mnlog = blaze::evaluate(neginf2zero(log((~lhs + ~rhs) * RT(0.5))));
    auto lhc = blaze::dot(~lhs, ~lhlog - mnlog);
    auto rhc = blaze::dot(~rhs, ~rhlog - mnlog);
    return RT(0.5) * (lhc + rhc);
}
template<typename VT, bool SO>
INLINE auto multinomial_jsd(const blaze::SparseVector<VT, SO> &lhs,
                            const blaze::SparseVector<VT, SO> &rhs)
{
    using RT = blz::ElementType_t<VT>;
    auto lhlog = blaze::log(lhs), rhlog = blaze::log(rhs);
    auto mnlog = blaze::evaluate(blaze::log((~lhs + ~rhs) * RT(0.5)));
    auto lhc = blaze::dot(~lhs, ~lhlog - mnlog);
    auto rhc = blaze::dot(~rhs, ~rhlog - mnlog);
    return RT(0.5) * (lhc + rhc);
}

template<typename FT, bool SO>
INLINE auto multinomial_bregman(const blaze::DenseVector<FT, SO> &lhs,
                                const blaze::DenseVector<FT, SO> &rhs,
                                const blaze::DenseVector<FT, SO> &lhlog,
                                const blaze::DenseVector<FT, SO> &rhlog)
{
    assert_all_nonzero(lhs);
    assert_all_nonzero(rhs);
    return blaze::dot(lhs, lhlog - rhlog);
}
template<typename FT, bool SO>
INLINE auto      poisson_bregman(const blaze::DenseVector<FT, SO> &lhs,
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
auto bhattacharyya_measure(const LHVec &lhs, const RHVec &rhs) {
    // Requires same storage.
    // TODO: generalize for different storage classes/transpose flags using DenseVector and SparseVector
    // base classes
    return sqrt(lhs * rhs);
}

template<typename LHVec, typename RHVec>
auto bhattacharyya_metric(const LHVec &lhs, const RHVec &rhs) {
    // Comaniciu, D., Ramesh, V. & Meer, P. (2003). Kernel-based object tracking.IEEE Transactionson Pattern Analysis and Machine Intelligence,25(5), 564-577.
    // Proves that this extension is a valid metric
    // See http://www.cse.yorku.ca/~kosta/CompVis_Notes/bhattacharyya.pdf
    return std::sqrt(1. - bhattacharyya_measure(lhs, rhs));
}
template<typename LHVec, typename RHVec>
auto matusita_distance(const LHVec &lhs, const RHVec &rhs) {
    return sqrL2Dist(sqrt(lhs), sqrt(rhs));
}
template<typename...Args>
INLINE decltype(auto) multinomial_jsm(Args &&...args) {
    using blaze::sqrt;
    using std::sqrt;
    return sqrt(multinomial_jsd(std::forward<Args>(args)...));
}

template<typename VT, typename VT2, bool SO>
inline auto s2jsd(const blz::Vector<VT, SO> &lhs, const blaze::Vector<VT2, SO> &rhs) {
    // Approximate jsd function for use in LSH tables.
    return std::sqrt(blz::sum(blz::pow(~lhs - ~rhs, 2) / (~lhs + ~rhs)) * ElementType_t<VT>(0.5));
}


template<typename VT, bool SO, typename VT2>
CommonType_t<ElementType_t<VT>, ElementType_t<VT2>>
network_p_wasserstein(const blz::DenseVector<VT, SO> &x, const blz::DenseVector<VT2, SO> &y, double p=1., size_t maxiter=10000)
{
    auto &xref = ~x;
    auto &yref = ~y;
    const size_t sz = xref.size();
    size_t nl = nonZeros(xref), nr = nonZeros(~y);
    using FT = CommonType_t<ElementType_t<VT>, ElementType_t<VT2>>;

    using namespace lemon;
    using Digraph = lemon::FullBipartiteDigraph;
    Digraph di(nl, nr);
    NetworkSimplexSimple<Digraph, FT, FT, unsigned, fgc::shared::flat_hash_map> net(di, true, nl + nr, nl * nr);
    DV<FT> weights(nl + nr);
    DV<unsigned> indices(nl + nr);
    size_t i = 0;
    for(size_t ii = 0; ii < sz; ++ii) {
        if(xref[ii] > 0)
            weights[i] = xref[ii], indices[i] = xref[ii], ++i;
    }
    for(size_t ii = 0; ii < sz; ++ii) {
        if(yref[ii] > 0)
            weights[i] = -yref[ii], indices[i] = yref[ii], ++i;
    }
    auto func = [p](auto x, auto y) {
        auto ret = x - y;
        if(p == 1) ret = std::abs(ret);
        else if(p == 2.) ret = ret * ret;
        else ret = std::pow(ret, p);
        return ret;
    };
    net.supplyMap(weights.data(), nl, weights.data() + nl, nr);
    {
        const auto jptr = &weights[nl];
        for(unsigned i = 0; i < nl; ++i) {
            auto arcid = i * nl;
            for(unsigned j = 0; j < nl; ++j) {
                net.setCost(di.arcFromId(arcid++, func(weights[i], jptr[j])));
            }
        }
    }
    int rc = net.run();
    if(rc != (int)net.OPTIMAL) {
        std::fprintf(stderr, "[%s:%s:%d] Warning: something went wrong in network simplex\n", __PRETTY_FUNCTION__, __FILE__, __LINE__);
    }

    FT ret(0);
    OMP_PRAGMA("omp parallel for reduction(+:ret)")
    for(size_t i = 0; i < nl; ++i) {
        for(size_t j = 0; j < nr; ++j)
           ret += net.flow(i * nr + j) * func(weights[i], weights[sz + j]);
    }
    return ret;
}

template<typename VT, bool SO, typename VT2>
CommonType_t<ElementType_t<VT>, ElementType_t<VT2>>
network_p_wasserstein(const blz::SparseVector<VT, SO> &x, const blz::SparseVector<VT2, SO> &y, double p=1., size_t maxiter=100)
{
    auto &xref = ~x;
    const size_t sz = xref.size();
    size_t nl = nonZeros(xref), nr = nonZeros(~y);
    using FT = CommonType_t<ElementType_t<VT>, ElementType_t<VT2>>;

    using namespace lemon;
	typedef lemon::FullBipartiteDigraph Digraph;
    Digraph di(nl, nr);
    NetworkSimplexSimple<Digraph, FT, FT, unsigned> net(di, true, nl + nr, nl * nr, maxiter);
    DV<FT> weights(nl + nr);
    DV<unsigned> indices(nl + nr);
    size_t i = 0;
    for(const auto &pair: xref)
        weights[i] = pair.value(), indices[i] = pair.index(), ++i;
    for(const auto &pair: ~y)
        weights[i] = -pair.value(), indices[i] = pair.index(), ++i; // negative weight
    auto func = [p](auto x, auto y) {
        auto ret = x - y;
        if(p == 1) ret = std::abs(ret);
        else if(p == 2.) ret = ret * ret;
        else ret = std::pow(ret, p);
        return ret;
    };
    net.supplyMap(weights.data(), nl, weights.data() + nl, nr);
    {
        const auto jptr = &weights[nl];
        for(unsigned i = 0; i < nl; ++i) {
            auto arcid = i * nl;
            for(unsigned j = 0; j < nl; ++j) {
                net.setCost(di.arcFromId(arcid++), func(weights[i], jptr[j]));
            }
        }
    }
    int rc = net.run();
    if(rc != (int)net.OPTIMAL) {
        std::fprintf(stderr, "[%s:%s:%d] Warning: something went wrong in network simplex\n", __PRETTY_FUNCTION__, __FILE__, __LINE__);
    }
    FT ret(0);
    OMP_PRAGMA("omp parallel for reduction(+:ret)")
    for(size_t i = 0; i < nl; ++i) {
        for(size_t j = 0; j < nr; ++j)
           ret += net.flow(i * nr + j) * func(weights[i], weights[sz + j]);
    }
    return ret;
}

template<typename VT, bool SO, typename VT2, typename CT=CommonType_t<ElementType_t<VT>, ElementType_t<VT2>>>
CT scipy_p_wasserstein(const blz::SparseVector<VT, SO> &x, const blz::SparseVector<VT2, SO> &y, double p=1.) {
    auto &xr = ~x;
    auto &yr = ~y;
    const size_t sz = xr.size();
    std::unique_ptr<uint32_t[]> ptr(new uint32_t[sz * 2]);
    auto xptr = ptr.get(), yptr = ptr.get() + sz;
    std::iota(xptr, yptr, 0u);
    std::iota(yptr, yptr + sz, 0u);
    pdqsort(xptr, yptr, [&](uint32_t p, uint32_t q) {return xr[p] < xr[q];});
    pdqsort(yptr, yptr + sz, [&](uint32_t p, uint32_t q) {return yr[p] < yr[q];});
    auto xconv = [&](auto x) {return xr[x];};
    auto yconv = [&](auto y) {return yr[y];};
    blz::DynamicVector<typename VT::ElementType, SO> all(2 * sz);
    std::merge(boost::make_transform_iterator(xptr, xconv), boost::make_transform_iterator(yptr, xconv),
               boost::make_transform_iterator(yptr, yconv), boost::make_transform_iterator(yptr + sz, yconv), all.begin());
    const size_t deltasz = 2 * sz - 1;
    blz::DynamicVector<typename VT::ElementType, SO> deltas(deltasz);
    std::adjacent_difference(all.begin(), all.end(), deltas.begin());
    auto fill_cdf = [&](auto datptr, const auto &datvec) -> blz::DynamicVector<typename VT::ElementType>
    {
        // Faster to do one linear scan than n binary searches
        blz::DynamicVector<typename VT::ElementType> ret(deltasz);
        for(size_t offset = 0, i = 0; i < ret.size(); ret[i++] = offset) {
            assert(i < all.size());
            while(offset < sz && datvec[datptr[offset]] < all[i]) {
                assert(&datptr[offset] < xptr + (2 * sz));
                ++offset;
            }
        }
        return ret;
    };
    auto cdfx = fill_cdf(xptr, xr);
    auto cdfy = fill_cdf(yptr, yr);
    CommonType_t<ElementType_t<VT>, ElementType_t<VT2>> ret;
    if(p == 1.)
        ret = dot(blz::abs(cdfx - cdfy), deltas) / sz;
    else {
       const auto szinv = 1. / sz;
       cdfx *= szinv; cdfy *= szinv;
       if(p == 2)
           ret = std::sqrt(dot(blz::pow(cdfx - cdfy, 2), deltas));
       else
           ret = std::pow(dot(blz::pow(blz::abs(cdfx - cdfy), p), deltas), 1. / p);
    }
    return ret;
}

template<typename VT, bool SO, typename VT2, typename CT=CommonType_t<ElementType_t<VT>, ElementType_t<VT2>>>
CT scipy_p_wasserstein(const blz::DenseVector<VT, SO> &x, const blz::DenseVector<VT2, SO> &y, double p=1.) {
    auto &xr = ~x;
    auto &yr = ~y;
    const size_t sz = xr.size();
    std::unique_ptr<uint32_t[]> ptr(new uint32_t[sz * 2]);
    auto xptr = ptr.get(), yptr = ptr.get() + sz;
    std::iota(xptr, yptr, 0u);
    std::iota(yptr, yptr + sz, 0u);
    pdqsort(xptr, yptr, [&](uint32_t p, uint32_t q) {return xr[p] < xr[q];});
    pdqsort(yptr, yptr + sz, [&](uint32_t p, uint32_t q) {return yr[p] < yr[q];});
    auto xconv = [&](auto x) {return xr[x];};
    auto yconv = [&](auto y) {return yr[y];};
    blz::DynamicVector<typename VT::ElementType, SO> all(2 * sz);
    std::merge(boost::make_transform_iterator(xptr, xconv), boost::make_transform_iterator(yptr, xconv),
               boost::make_transform_iterator(yptr, yconv), boost::make_transform_iterator(yptr + sz, yconv), all.begin());
    const size_t deltasz = 2 * sz - 1;
    blz::DynamicVector<typename VT::ElementType, SO> deltas(deltasz);
    std::adjacent_difference(all.begin(), all.end(), deltas.begin());
    assert(std::is_sorted(xptr, yptr,  [&](uint32_t p, uint32_t q) {return xr[p] < xr[q];}));
    auto fill_cdf = [&](auto datptr, const auto &datvec) -> blz::DynamicVector<typename VT::ElementType>
    {
        // Faster to do one linear scan than n binary searches
        blz::DynamicVector<typename VT::ElementType> ret(deltasz);
        for(size_t offset = 0, i = 0; i < ret.size(); ret[i++] = offset) {
            assert(i < all.size());
            while(offset < sz && datvec[datptr[offset]] < all[i]) {
                assert(&datptr[offset] < xptr + (2 * sz));
                ++offset;
            }
        }
        return ret;
    };
    auto cdfx = fill_cdf(xptr, xr);
    auto cdfy = fill_cdf(yptr, yr);
    CommonType_t<ElementType_t<VT>, ElementType_t<VT2>> ret;
    if(p == 1.)
        ret = dot(blz::abs(cdfx - cdfy), deltas) / sz;
    else {
       const auto szinv = 1. / sz;
       cdfx *= szinv; cdfy *= szinv;
       if(p == 2)
           ret = std::sqrt(dot(blz::pow(cdfx - cdfy, 2), deltas));
       else
           ret = std::pow(dot(blz::pow(blz::abs(cdfx - cdfy), p), deltas), 1. / p);
    }
    return ret;
}

template<typename VT, bool SO, typename VT2, typename CT=CommonType_t<ElementType_t<VT>, ElementType_t<VT2>>>
CT p_wasserstein(const blz::DenseVector<VT, SO> &x, const blz::DenseVector<VT2, SO> &y, double p=1.) {
    return scipy_p_wasserstein(x, y, p);
}

template<typename VT, bool SO, typename VT2, typename CT=CommonType_t<ElementType_t<VT>, ElementType_t<VT2>>>
CT p_wasserstein(const blz::SparseVector<VT, SO> &x, const blz::SparseVector<VT2, SO> &y, double p=1.) {
    return scipy_p_wasserstein(x, y, p);
}

template<typename VT, bool SO, typename VT2>
auto wasserstein_p2(const blz::Vector<VT, SO> &x, const blz::Vector<VT2, SO> &y) {
    return p_wasserstein(~x, ~y, 2.);
}
template<typename VT, bool SO, typename VT2>
auto wasserstein_p2(const blz::Vector<VT, SO> &x, const blz::Vector<VT2, !SO> &y) {
    return p_wasserstein(~x, trans(~y), 2.);
}

template<typename VT, typename VT2, bool SO>
auto discrete_total_variation_distance(const blz::Vector<VT, SO> &lhs, const blz::Vector<VT2, SO> &rhs) {
    return ElementType_t<VT>(0.5) * blz::l1Norm(~lhs - ~rhs);
}

#if 0
template<typename VT, typename VT2, bool SO>
auto witten_poisson_dissimilarity(const blz::Vector<VT, SO> &lhs, const blz::Vector<VT2, SO> &rhs, ) {

}
#endif

} // distance

} // namespace blz

#endif // FGC_DISTANCE_AND_MEANING_H__
