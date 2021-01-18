#ifndef FGC_DISTANCE_AND_MEANING_H__
#define FGC_DISTANCE_AND_MEANING_H__
#include <vector>
#include <iostream>
#include <set>


#include "minicore/util/blaze_adaptor.h"

#ifndef BOOST_NO_AUTO_PTR
#define BOOST_NO_AUTO_PTR 1
#endif

#include "boost/iterator/transform_iterator.hpp"

namespace minicore {

namespace distance {

using namespace blz;



enum DissimilarityMeasure {
    L1,
    L2,
    SQRL2,
    JSM, // Multinomial Jensen-Shannon Metric
    JSD, // Multinomial Jensen-Shannon Divergence
    MKL, // Multinomial KL Divergence
    HELLINGER,
    BHATTACHARYYA_METRIC,
    BHATTACHARYYA_DISTANCE,
    TOTAL_VARIATION_DISTANCE,
    LLR,
    REVERSE_MKL,
    UWLLR, /* Unweighted Log-likelihood Ratio.
            * Specifically, this is the D_{JSD}^{\lambda}(x, y),
            * where \lambda = \frac{N_p}{N_p + N_q}
            *
            */
    ITAKURA_SAITO, // \sum_{i=1}^D[\frac{a_i}{b_i} - \log{\frac{a_i}{b_i}} - 1]
    REVERSE_ITAKURA_SAITO, // Reverse I-S
    COSINE_DISTANCE,             // Cosine distance
    PROBABILITY_COSINE_DISTANCE, // Probability distribution cosine distance
    COSINE_SIMILARITY,
    PROBABILITY_COSINE_SIMILARITY,
    DOT_PRODUCT_SIMILARITY,
    PROBABILITY_DOT_PRODUCT_SIMILARITY,
    ORACLE_METRIC,
    ORACLE_PSEUDOMETRIC,
    SYMMETRIC_ITAKURA_SAITO,
    REVERSE_SYMMETRIC_ITAKURA_SAITO,
    SRLRT,
    SRULRT,
    WLLR = LLR, // Weighted Log-likelihood Ratio, now equivalent to the LLR
    TVD = TOTAL_VARIATION_DISTANCE,
    PSD = JSD, // Poisson JSD, but algebraically equivalent
    PSM = JSM,
    IS=ITAKURA_SAITO,
    SIS=SYMMETRIC_ITAKURA_SAITO,
    RSYMMETRIC_ITAKURA_SAITO=REVERSE_SYMMETRIC_ITAKURA_SAITO,
    RSIS=REVERSE_SYMMETRIC_ITAKURA_SAITO,
};

static constexpr inline bool msr_is_normalized(DissimilarityMeasure msr) {
    switch(msr) {
#if 0
        case IS: case SIS: case RSIS: case REVERSE_ITAKURA_SAITO: case TVD:
        case MKL: case REVERSE_MKL: case COSINE_DISTANCE: case COSINE_SIMILARITY:
        case HELLINGER: case BHATTACHARYYA_METRIC: case BHATTACHARYYA_DISTANCE:
        case JSM: case JSD:
            return true;
#endif

        case LLR:
        case SRLRT: case L1: case L2: case SQRL2: case UWLLR: case SRULRT:
        default:
        return false;
    }
}

inline namespace detail {
/*
 *
 * Traits for each divergence, whether they need:
 * 1. Cached data
 *     1. Logs  (needs_logs)
 *     2. Sqrts (needs_sqrt)
 *     3. L2 norms (needs_l2_cache)
 * 2. Whether the measure satisfies:
 *     1. Being a Bregman divergence (is_bregman)
 *     2. Being a distance metric (satisfies_metric)
 *     3. Being a rho-approximate distance metric (satisfies_rho_metric)
 *     4. D^2 sampling requirements
 *
 * Generating an approximate solution is fastest/easiest for those satisfying d2,
 * where d^2 sampling can provide an approximate solution in linear time.
 * If that is unavailable, then Jain-Vazirani/local search, optionally with
 * Thorup-based facility location sampling, should be used for an initial approximation.
 *
 * After an approximate solution is generated, a coreset sampler can be built.
 * This should be VARADARAJAN_XIAO for constant factor approximations under metrics,
 * LUCIC_BACHEM_KRAUSE for Bregman divergences,
 * and BRAVERMAN_FELDMAN_LANG or FELDMAN_LANGBERG for bicriteria approximations.
 *
 * After generating a coreset, it should be optimized.
 * For Bregman divergences (soft or hard) and SQRL2, this is done with EM in a loop of Lloyd's.
 * For LLR/UWLLR, this should be done in the same fashion.
 * For L1, EM should be run, but the mean should be the componentwise median instead.
 * For TOTAL_VARIATION_DISTANCE, it the L1 algorithm should be run on the normalized values.
 * For all other distance measures, Jain-Vazirani and/or local search should be run.
 *
 */

static constexpr INLINE bool is_bregman(DissimilarityMeasure d)  {
    switch(d) {
        case JSD: case MKL: case ITAKURA_SAITO:
        case REVERSE_MKL: case REVERSE_ITAKURA_SAITO:
        case SYMMETRIC_ITAKURA_SAITO: case REVERSE_SYMMETRIC_ITAKURA_SAITO:
        case SQRL2: case SRLRT: case SRULRT:
        return true;
        default: ;
    }
    return false;
}
static constexpr INLINE bool satisfies_d2(DissimilarityMeasure d) {
    return d == LLR || d == UWLLR || is_bregman(d);
}
static constexpr INLINE bool satisfies_metric(DissimilarityMeasure d) {
    switch(d) {
        case L1:
        case L2:
        case JSM:
        case BHATTACHARYYA_METRIC:
        case TOTAL_VARIATION_DISTANCE:
        case HELLINGER:
        case ORACLE_METRIC:
        case SRLRT: case SRULRT: // Not sure, but probably
            return true;
        default: ;
    }
    return false;
}
static constexpr INLINE bool satisfies_rho_metric(DissimilarityMeasure d) {
    if(satisfies_metric(d)) return true;
    switch(d) {
        case SQRL2: // rho = 2
        // These three don't, technically, but using a prior can force it to follow it on real data
        case ORACLE_PSEUDOMETRIC:
        case LLR: case UWLLR: case SYMMETRIC_ITAKURA_SAITO: case REVERSE_SYMMETRIC_ITAKURA_SAITO:
            return true;
        default:;
    }
    return false;
}

static constexpr INLINE bool needs_logs(DissimilarityMeasure d)  {
    switch(d) {
        case JSM: case JSD: case MKL: case LLR: case ITAKURA_SAITO:
        case SRLRT: case SRULRT:
        case REVERSE_MKL: case UWLLR: case REVERSE_ITAKURA_SAITO: case SYMMETRIC_ITAKURA_SAITO: case REVERSE_SYMMETRIC_ITAKURA_SAITO: return true;
        default: break;
    }
    return false;
}

static constexpr INLINE bool use_scaled_centers(DissimilarityMeasure measure) {
    // Whether centers should be produced as being not normalized by row sums
    // compared to default behavior
    switch(measure) {
        case LLR:
        case UWLLR:
        case L1:
        case SQRL2:
        case L2:
        case COSINE_DISTANCE:
        case SYMMETRIC_ITAKURA_SAITO:
        case RSYMMETRIC_ITAKURA_SAITO:
           return true;
        default: return false;
    }
}

static constexpr INLINE bool is_probability(DissimilarityMeasure d)  {
    switch(d) {
        case TOTAL_VARIATION_DISTANCE: case BHATTACHARYYA_METRIC: case BHATTACHARYYA_DISTANCE:
        case HELLINGER:
        case MKL: case REVERSE_MKL:
        case PROBABILITY_COSINE_DISTANCE: case PROBABILITY_DOT_PRODUCT_SIMILARITY:
        case ITAKURA_SAITO: case REVERSE_ITAKURA_SAITO:
        case SYMMETRIC_ITAKURA_SAITO:
        case RSYMMETRIC_ITAKURA_SAITO:
        return true;
        default: break;
    }
    return false;
}

static constexpr INLINE bool needs_l2_cache(DissimilarityMeasure d) {
    return d == COSINE_DISTANCE;
}

static constexpr bool expects_nonnegative(DissimilarityMeasure measure) {
    switch(measure) {
        case L1: case L2: case SQRL2:
        case COSINE_DISTANCE: case COSINE_SIMILARITY:
        case PROBABILITY_COSINE_DISTANCE: case PROBABILITY_COSINE_SIMILARITY:
        case DOT_PRODUCT_SIMILARITY:

        default: // Unexpected, but will assume it's required.
        case JSM: case JSD: case MKL: case HELLINGER: case BHATTACHARYYA_METRIC:
        case BHATTACHARYYA_DISTANCE: case TOTAL_VARIATION_DISTANCE: case LLR:
        case REVERSE_MKL: case ITAKURA_SAITO: case REVERSE_ITAKURA_SAITO:
        case PROBABILITY_DOT_PRODUCT_SIMILARITY:
        case SYMMETRIC_ITAKURA_SAITO:
        case RSYMMETRIC_ITAKURA_SAITO:
        return true;
    }
}

static constexpr INLINE bool is_dissimilarity(DissimilarityMeasure d) {
    switch(d) {
        case DOT_PRODUCT_SIMILARITY: case PROBABILITY_DOT_PRODUCT_SIMILARITY:
        case COSINE_SIMILARITY:      case PROBABILITY_COSINE_SIMILARITY:
            return false;
        default: ;
    }
    return true;
}


static constexpr INLINE bool needs_probability_l2_cache(DissimilarityMeasure d) {
    return d == PROBABILITY_COSINE_DISTANCE;
}

static constexpr INLINE bool  needs_sqrt(DissimilarityMeasure d) {
    return d == HELLINGER || d == BHATTACHARYYA_METRIC || d == BHATTACHARYYA_DISTANCE;
}

static constexpr INLINE bool is_symmetric(DissimilarityMeasure d) {
    switch(d) {
        case L1: case L2: case HELLINGER: case BHATTACHARYYA_DISTANCE: case BHATTACHARYYA_METRIC:
        case JSD: case JSM: case LLR: case UWLLR: case SQRL2: case TOTAL_VARIATION_DISTANCE:
        case COSINE_DISTANCE: case COSINE_SIMILARITY:
        case PROBABILITY_COSINE_DISTANCE: case PROBABILITY_COSINE_SIMILARITY:
        case SYMMETRIC_ITAKURA_SAITO:
        case RSYMMETRIC_ITAKURA_SAITO:
        case SRULRT: case SRLRT:
            return true;
        default: ;
    }
    return false;
}

template<typename VT, bool TF, typename VT2, typename CVT=VT, bool TF2=TF>
void set_cache(const blz::Vector<VT, TF> &src, blz::Vector<VT2, TF> &dest, DissimilarityMeasure d, blz::Vector<CVT, TF2> *cp=nullptr) {
    if(!cp) {
        if(needs_logs(d)) {
            if(is_probability(d))
                *dest = neginf2zero(log(*src));
            else
                *dest = neginf2zero(log(*src / blaze::sum(*src)));
            return;
        }
        if(needs_sqrt(d)) {
            *dest = sqrt(*src);
            return;
        }
    } else {
        throw std::runtime_error("Not implemented: setting cache for the instance of a nonzero prior");
    }
}

static constexpr INLINE const char *prob2str(DissimilarityMeasure d) {
    switch(d) {
        case BHATTACHARYYA_DISTANCE: return "BHATTACHARYYA_DISTANCE";
        case BHATTACHARYYA_METRIC: return "BHATTACHARYYA_METRIC";
        case HELLINGER: return "HELLINGER";
        case JSD: return "JSD";
        case JSM: return "JSM";
        case L1: return "L1";
        case L2: return "L2";
        case LLR: return "LLR";
        case UWLLR: return "UWLLR";
        case ITAKURA_SAITO: return "ITAKURA_SAITO";
        case MKL: return "MKL";
        case REVERSE_MKL: return "REVERSE_MKL";
        case REVERSE_ITAKURA_SAITO: return "REVERSE_ITAKURA_SAITO";
        case SQRL2: return "SQRL2";
        case TOTAL_VARIATION_DISTANCE: return "TOTAL_VARIATION_DISTANCE";
        case COSINE_DISTANCE: return "COSINE_DISTANCE";
        case PROBABILITY_COSINE_DISTANCE: return "PROBABILITY_COSINE_DISTANCE";
        case COSINE_SIMILARITY: return "COSINE_SIMILARITY";
        case PROBABILITY_COSINE_SIMILARITY: return "PROBABILITY_COSINE_SIMILARITY";
        case ORACLE_METRIC: return "ORACLE_METRIC";
        case ORACLE_PSEUDOMETRIC: return "ORACLE_PSEUDOMETRIC";
        case SRULRT: return "SRULRT";
        case SRLRT: return "SRLRT";
        case SYMMETRIC_ITAKURA_SAITO: return "SYMMETRIC_ITAKURA_SAITO";
        case RSYMMETRIC_ITAKURA_SAITO: return "RSYMMETRIC_ITAKURA_SAITO";
        default: return "INVALID TYPE";
    }
}
static constexpr INLINE const char *msr2str(DissimilarityMeasure d) {
    return prob2str(d);
}
static constexpr INLINE const char *prob2desc(DissimilarityMeasure d) {
    switch(d) {
        case BHATTACHARYYA_DISTANCE: return "Bhattacharyya distance: -log(dot(sqrt(x) * sqrt(y)))";
        case BHATTACHARYYA_METRIC: return "Bhattacharyya metric: sqrt(1 - BhattacharyyaSimilarity(x, y))";
        case HELLINGER: return "Hellinger Distance: sqrt(sum((sqrt(x) - sqrt(y))^2))/2";
        case JSD: return "Jensen-Shannon Divergence for Poisson and Multinomial models, for which they are equivalent";
        case JSM: return "Jensen-Shannon Metric, known as S2JSD and the Endres metric, for Poisson and Multinomial models, for which they are equivalent";
        case L1: return "L1 distance";
        case L2: return "L2 distance";
        case LLR: return "Log-likelihood Ratio under the multinomial model";
        case UWLLR: return "Unweighted Log-likelihood Ratio. This is effectively the Generalized Jensen-Shannon Divergence with lambda parameter corresponding to the fractional contribution of counts in the first observation. This is symmetric, unlike the G_JSD, because the parameter comes from the counts.";
        case MKL: return "Multinomial KL divergence";
        case REVERSE_MKL: return "Reverse Multinomial KL divergence";
        case SQRL2: return "Squared L2 Norm";
        case TOTAL_VARIATION_DISTANCE: return "Total Variation Distance: 1/2 sum_{i in D}(|x_i - y_i|)";
        case ITAKURA_SAITO: return "Itakura-Saito divergence, a Bregman divergence [sum((a / b) - log(a / b) - 1 for a, b in zip(A, B))]";
        case SYMMETRIC_ITAKURA_SAITO: return "Symmetrized Itakura-Saito divergence. IS is a [sum((a / b) - log(a / b) - 1 for a, b in zip(A, B))], while SIS is .5 * (IS(a, (a + b) / 2) + IS(b, (a + b) / 2)), analogous to JSD";
        case RSYMMETRIC_ITAKURA_SAITO: return "Reversed symmetrized Itakura-Saito divergence. IS is a [sum((a / b) - log(a / b) - 1 for a, b in zip(A, B))], while SIS is .5 * (IS((a + b) / 2, a) + IS((a + b) / 2, b)), analogous to JSD";
        case REVERSE_ITAKURA_SAITO: return "Reversed Itakura-Saito divergence, a Bregman divergence";
        case COSINE_DISTANCE: return "Cosine distance: arccos(\\frac{A \\cdot B}{|A|_2 |B|_2}) / pi";
        case PROBABILITY_COSINE_DISTANCE: return "Cosine distance of the probability vectors: arccos(\\frac{A \\cdot B}{|A|_2 |B|_2}) / pi";
        case COSINE_SIMILARITY: return "Cosine similarity: \\frac{A \\cdot B}{|A|_2 |B|_2}";
        case PROBABILITY_COSINE_SIMILARITY: return "Cosine similarity of the probability vectors: \\frac{A \\cdot B}{|A|_2 |B|_2}";
        case ORACLE_METRIC: return "Placeholder for oracle metrics, allowing us to use DissimilarityMeasure in other situations";
        case ORACLE_PSEUDOMETRIC: return "Placeholder for oracle pseudometrics";
        case SRULRT: return "Square root of UWLLR, unweighted log likelihood ratio test; likely a metric: related to the JSM and Generalized JSD";
        case SRLRT: return "Square root of LRT, the log likelihood ratio test; likely a metric: related to the JSM and Generalized JSD";
        default: return prob2str(d);
    }
}

static constexpr DissimilarityMeasure USABLE_MEASURES []  {
    L1,
    L2,
    SQRL2,
    JSM,
    JSD,
    MKL,
    HELLINGER,
    BHATTACHARYYA_METRIC,
    BHATTACHARYYA_DISTANCE,
    TOTAL_VARIATION_DISTANCE,
    LLR,
    UWLLR,
    REVERSE_MKL,
    ITAKURA_SAITO,
    REVERSE_ITAKURA_SAITO,
    COSINE_DISTANCE,
    COSINE_SIMILARITY,
    SYMMETRIC_ITAKURA_SAITO,
    RSYMMETRIC_ITAKURA_SAITO,
    SRLRT,
    SRULRT
    // Absent:
    // PROBABILITY_COSINE_DISTANCE/PROBABILITY_COSINE_SIMILARITY -- extensions to this space are not complete.
    // ORACLE_METRIC/ORACLE_PSEUDOMETRIC, as they are placeholders
};

static inline DissimilarityMeasure str2msr(const std::string &s) {
    for(const auto sm: USABLE_MEASURES) if(s == prob2str(sm)) return sm;
    throw std::invalid_argument(s);
}

static void print_measures() {
    for(const auto measure: USABLE_MEASURES) {
        std::fprintf(stderr, "Code: %d. Description: '%s'. Short name: '%s'\n", measure, prob2desc(measure), prob2str(measure));
    }
}

static constexpr bool is_valid_measure(DissimilarityMeasure measure) {
    switch(measure) {
        case L1: case L2: case SQRL2: case JSM: case JSD: case MKL:
        case HELLINGER: case BHATTACHARYYA_METRIC:
        case BHATTACHARYYA_DISTANCE: case TOTAL_VARIATION_DISTANCE:
        case UWLLR: case LLR: case REVERSE_MKL: case REVERSE_ITAKURA_SAITO:
        case ITAKURA_SAITO: case COSINE_DISTANCE: case PROBABILITY_COSINE_DISTANCE:
        case DOT_PRODUCT_SIMILARITY: case PROBABILITY_DOT_PRODUCT_SIMILARITY:
        case ORACLE_METRIC: case ORACLE_PSEUDOMETRIC:
        case SYMMETRIC_ITAKURA_SAITO:
        case RSYMMETRIC_ITAKURA_SAITO:
        case SRLRT: case SRULRT:
        return true;
        default: ;
    }
    return false;
}


enum RestartMethodPol {
    RESTART_D2,
    RESTART_GREEDY,
    RESTART_RANDOM
};


} // detail

inline namespace constants {

template<typename FT>
static constexpr FT RSIS_OFFSET = 0.1931471805599453;
template<typename FT>
static constexpr FT SIS_OFFSET = -.6931471805599453;

}


template<typename FT, bool SO>
auto logsumexp(const blaze::DenseVector<FT, SO> &x) {
    const auto maxv = blaze::max(*x);
    return maxv + std::log(blaze::sum(blaze::exp(*x - maxv)));
}

template<typename FT, bool SO>
auto logsumexp(const blaze::SparseVector<FT, SO> &x) {
    auto maxv = blaze::max(*x);
    auto s = 0.;
    for(const auto p: *x) {
        s += std::exp(p.value() - maxv); // Sum over sparse elements
    }
    s += ((*x).size() - nonZeros(*x)) * std::exp(-maxv);  // Handle the ones we skipped
    return maxv + std::log(s);
}

template<typename VT1, typename VT2, typename Scalar, bool TF>
auto logsumexp(const SVecScalarMultExpr<SVecSVecAddExpr<VT1, VT2, TF>, Scalar, TF> &exp) {
    // Specifically for calculating the logsumexp of the mean of the two sparse vectors.
    // SVecScalarMultExpr doesn't provide a ConstIterator, so we're rolling our own specialized function.
    auto maxv = blaze::max(*exp), mmax = -maxv;
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
    s += ((*exp).size() - nnz) * std::exp(-maxv);  // Handle the ones we skipped
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
    auto mn = (*lhs + *rhs) * RT(0.5);
    auto mnlog = blaze::evaluate(blaze::neginf2zero(blaze::log(*mn)));
    auto lhc = blaze::dot(*lhs, *lhlog - mnlog);
    auto rhc = blaze::dot(*rhs, *rhlog - mnlog);
    return RT(0.5) * (lhc + rhc);
}

template<typename VT, typename VT2, bool SO>
INLINE auto multinomial_jsd(const blaze::SparseVector<VT, SO> &lhs,
                            const blaze::SparseVector<VT, SO> &rhs,
                            const blaze::SparseVector<VT2, SO> &lhlog,
                            const blaze::SparseVector<VT2, SO> &rhlog)
{
    using RT = blz::CommonType_t<blz::ElementType_t<VT>, blz::ElementType_t<VT2>>;
    auto mnlog = blaze::evaluate(blaze::log((*lhs + *rhs) * RT(0.5)));
    auto lhc = blaze::dot(*lhs, *lhlog - mnlog);
    auto rhc = blaze::dot(*rhs, *rhlog - mnlog);
    return RT(0.5) * (lhc + rhc);
}
template<typename VT, bool SO>
INLINE auto multinomial_jsd(const blaze::DenseVector<VT, SO> &lhs,
                            const blaze::DenseVector<VT, SO> &rhs)
{
    using RT = blz::ElementType_t<VT>;
    auto lhlog = blaze::evaluate(neginf2zero(log(lhs)));
    auto rhlog = blaze::evaluate(neginf2zero(log(rhs)));
    auto mnlog = blaze::evaluate(neginf2zero(log((*lhs + *rhs) * RT(0.5))));
    auto lhc = blaze::dot(*lhs, *lhlog - mnlog);
    auto rhc = blaze::dot(*rhs, *rhlog - mnlog);
    return RT(0.5) * (lhc + rhc);
}
template<typename VT, typename VT2, bool SO>
INLINE auto multinomial_jsd(const blaze::Vector<VT, SO> &lhs,
                            const blaze::Vector<VT2, SO> &rhs)
{
    using RT = blz::ElementType_t<VT>;
    auto lhlog = blaze::evaluate(neginf2zero(log(*lhs)));
    auto rhlog = blaze::evaluate(neginf2zero(log(*rhs)));
    auto mnlog = blaze::evaluate(neginf2zero(log((*lhs + *rhs) * RT(0.5))));
    auto lhc = blaze::dot(*lhs, *lhlog - mnlog);
    auto rhc = blaze::dot(*rhs, *rhlog - mnlog);
    return RT(0.5) * (lhc + rhc);
}
template<typename VT, bool SO>
INLINE auto multinomial_jsd(const blaze::SparseVector<VT, SO> &lhs,
                            const blaze::SparseVector<VT, SO> &rhs)
{
    using RT = blz::ElementType_t<VT>;
    auto lhlog = blaze::log(lhs), rhlog = blaze::log(rhs);
    auto mnlog = blaze::evaluate(blaze::log((*lhs + *rhs) * RT(0.5)));
    auto lhc = blaze::dot(*lhs, *lhlog - mnlog);
    auto rhc = blaze::dot(*rhs, *rhlog - mnlog);
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

static constexpr const char *prior2str(Prior p) {
    switch(p) {
        case NONE: return "NONE";
        case DIRICHLET: return "DIRICHLET";
        case GAMMA_BETA: return "GAMMA_BETA";
        case FEATURE_SPECIFIC_PRIOR: return "FEATURE_SPECIFIC_PRIOR";
        default: return "unknown";
    }
}

static constexpr const char *prior2desc(Prior p) {
    switch(p) {
        case DIRICHLET: return "DIRICHLET: Uniform prior, smoothes divergences";
        case GAMMA_BETA: return "GAMMA_BETA: Uniform prior, soothes divergences, parametrized. Larger values make points more similar, the smaller makes points less similar";
        case NONE: return "NONE: no prior.";
        default: return prior2str(p);
    }
}


template<typename VT1, typename VT2, bool SO, bool OSO, typename CT=CommonType_t<ElementType_t<VT1>, ElementType_t<VT2>>, typename OFT>
CT cosine_similarity(const blz::Vector<VT1, SO> &x, const blz::Vector<VT2, OSO> &y, OFT xnorm, OFT ynorm) {
    return blz::dot(*x, *y) / (xnorm * ynorm);
}
template<typename VT1, typename VT2, bool SO, bool OSO, typename CT=CommonType_t<ElementType_t<VT1>, ElementType_t<VT2>>>
CT cosine_similarity(const blz::Vector<VT1, SO> &x, const blz::Vector<VT2, OSO> &y) {
    return blz::dot(*x, *y) / (blz::l2Norm(*x) * blz::l2Norm(*y));
}
template<typename VT1, typename VT2, bool SO, bool OSO, typename CT=CommonType_t<ElementType_t<VT1>, ElementType_t<VT2>>, typename OFT>
CT cosine_distance(const blz::Vector<VT1, SO> &x, const blz::Vector<VT2, OSO> &y, OFT xnorm, OFT ynorm) {
    static constexpr CT PI_INV = 1. / 3.14159265358979323846264338327950288;
    return std::acos(cosine_similarity(x, y, xnorm, ynorm)) * PI_INV;
}
template<typename VT1, typename VT2, bool SO, bool OSO, typename CT=CommonType_t<ElementType_t<VT1>, ElementType_t<VT2>>>
CT cosine_distance(const blz::Vector<VT1, SO> &x, const blz::Vector<VT2, OSO> &y) {
    static constexpr CT PI_INV = 1. / 3.14159265358979323846264338327950288;
    return std::acos(cosine_similarity(x, y, blz::l2Norm(*x), blz::l2Norm(*y))) * PI_INV;
}

template<typename VT1, typename VT2, bool TF>
auto bhattacharyya_measure(const blz::DenseVector<VT1, TF> &lhs, const blz::DenseVector<VT2, TF> &rhs) {
    // Requires same storage.
    // TODO: generalize for different storage classes/transpose flags using DenseVector and SparseVector
    // base classes
    return sum(sqrt(*lhs * *rhs));
}

template<typename LHVec, typename RHVec>
auto bhattacharyya_metric(const LHVec &lhs, const RHVec &rhs) {
    // Comaniciu, D., Ramesh, V. & Meer, P. (2003). Kernel-based object tracking.IEEE Transactionson Pattern Analysis and Machine Intelligence,25(5), 564-577.
    // Proves that this extension is a valid metric
    // See http://www.cse.yorku.ca/*kosta/CompVis_Notes/bhattacharyya.pdf
    return std::sqrt(1. - bhattacharyya_measure(lhs, rhs));
}
template<typename LHVec, typename RHVec>
auto bhattacharyya_distance(const LHVec &lhs, const RHVec &rhs) {
    // Comaniciu, D., Ramesh, V. & Meer, P. (2003). Kernel-based object tracking.IEEE Transactionson Pattern Analysis and Machine Intelligence,25(5), 564-577.
    // Proves that this extension is a valid metric
    // See http://www.cse.yorku.ca/*kosta/CompVis_Notes/bhattacharyya.pdf
    return -std::log(bhattacharyya_measure(lhs, rhs));
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
    return std::sqrt(blz::sum(blz::pow(*lhs - *rhs, 2) / (*lhs + *rhs)) * ElementType_t<VT>(0.5));
}


template<typename VT, bool SO, typename VT2, typename CT=CommonType_t<ElementType_t<VT>, ElementType_t<VT2>>>
CT scipy_p_wasserstein(const blz::SparseVector<VT, SO> &x, const blz::SparseVector<VT2, SO> &y, double p=1.) {
    auto &xr = *x;
    auto &yr = *y;
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
CT scipy_p_wasserstein(const blz::Vector<VT, SO> &x, const blz::Vector<VT2, SO> &y, double p=1.) {
    auto &xr = *x;
    auto &yr = *y;
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
CT p_wasserstein(const blz::Vector<VT, SO> &x, const blz::Vector<VT2, SO> &y, double p=1.) {
    return scipy_p_wasserstein(x, y, p);
}

template<typename VT, bool SO, typename VT2>
auto wasserstein_p2(const blz::Vector<VT, SO> &x, const blz::Vector<VT2, SO> &y) {
    return p_wasserstein(*x, *y, 2.);
}
template<typename VT, bool SO, typename VT2>
auto wasserstein_p2(const blz::Vector<VT, SO> &x, const blz::Vector<VT2, !SO> &y) {
    return p_wasserstein(*x, trans(*y), 2.);
}

template<typename VT, typename VT2, bool SO>
static INLINE auto discrete_total_variation_distance(const blz::Vector<VT, SO> &lhs, const blz::Vector<VT2, SO> &rhs) {
    return ElementType_t<CommonType_t<VT, VT2>>(0.5) * blz::l1Norm(*lhs - *rhs);
}

template<typename VT, typename VT2, bool SO>
static INLINE auto canberra_distance(const blz::DenseVector<VT, SO> &lhs, const blz::DenseVector<VT2, SO> &rhs) {
    const auto &lh(*lhs), &rh(*rhs);
    return blaze::sum(blaze::abs(lh - rh) / (blaze::abs(lh) + blaze::abs(rh)));
}

template<typename VT, typename VT2, bool SO, bool SO2>
static INLINE auto hellinger(const blz::Vector<VT, SO> &lhs, const blz::Vector<VT2, SO2> &rhs) {
    if constexpr(SO == SO2) {
        return l2Norm(sqrt(*lhs) - sqrt(*rhs));
    } else {
        return l2Norm(sqrt(*lhs) - trans(sqrt(*rhs)));
    }
}


} // distance

} // namespace minicore

namespace dist = minicore::distance;

#endif // FGC_DISTANCE_AND_MEANING_H__
