#ifndef FGC_CLUSTERING_H__
#define FGC_CLUSTERING_H__
#include "minocore/dist.h"
#include <cstdint>

namespace minocore {

namespace clustering {
using ClusteringEnumType = std::size_t;
using ce_t = ClusteringEnumType;

/*
 *
 * Problem classification. These are:
 * 1. Assignment
 * IE, is the assignment to cluster centers hard or soft?
 * 2. Center Origination
 * Are cluster centers Intrinsic or Extrinsic? (IE, are centers selected from input points or not?)
 * 3. Type of approximate solution.
 * If using coresets, which algorithm is used for an approximate solution?
 * BICRITERIA (for alpha-beta approximations, where a constant factor approximation with more than alpha centers is allowed)
 * CONSTANT_FACTOR (for the exact number of centers but with a constant factor approximation)
 * HEURISTIC (for a good enough solution, which we will treat as one of the above even though it isn't)
 * 4. Center sampling type.
 * When selecting centers to use as candidates in search, use:
 * Thorup sampling
 * Uniform sampling
 * D2/cost sampling
 * 5. Optimization technique
 * Metric k-median: use metric clustering techniques, such as Jain-Vazirani or local search
 * Expectation Maximization: Lloyd's algorithm or a variant
 * Gradient descent (will require autograd or similar)
 * Exhaustive search: combinatorial approximation
 * Black box: plugging into CPLEX or Gurobi
 */

enum Assignment: ce_t {
    HARD = 0,
    /* Assignment(x) = argmin_{c \in C}[d(c, x)]
     *
     */
    SOFT = 1,
    // Assignment(X, c) = \frac{c}{\sum_{c' \in C}[d(c', x)]}
    SOFTMAX = 2,
    /* Assignment(X, c) = \frac{e^{d(c, x)}}{sum_{c' \in C}[e^d(c', x)]}
     *                  = softmax(d(C, x))
     */
};
enum CenterOrigination: ce_t {
    INTRINSIC = 0,
    EXTRINSIC = 1,
};
enum ApproximateSolutionType: ce_t {
    BICRITERIA      = 0,
    CONSTANT_FACTOR = 1,
    HEURISTIC       = 2,
    RSVD            = 3,
};
enum CenterSamplingType: ce_t {
    THORUP_SAMPLING,
    D2_SAMPLING,
    UNIFORM_SAMPLING,
    GREEDY_SAMPLING,
    COST_SAMPLING = D2_SAMPLING,
};
enum OptimizationMethod: ce_t {
    METRIC_KMEDIAN,
    EXPECTATION_MAXIMIZATION,
    BLACK_BOX,
    GRADIENT_DESCENT,
    EXHAUSTIVE_SEARCH,
};

enum ClusteringBitfields: ce_t {
    ASN_BF_OFFSET = 3,
    HARD_BF = 1 << HARD,
    SOFT_BF = 1 << SOFT,
    SOFTMAX_BF = 1 << SOFTMAX,
    CO_BF_OFFSET = ASN_BF_OFFSET + 2
    INTRINSIC_BF = 1 << ASN_BF_OFFSET,
    EXTRINSIC_BF = 1 << (EXTRINSIC + ASN_BF_OFFSET),
    AS_BF_OFFSET = CO_BF_OFFSET + 4
    BICRITERIA_BF      = 1ull << (BICRITERIA + CO_BF_OFFSET),
    CONSTANT_FACTOR_BF = 1ull << (CONSTANT_FACTOR + CO_BF_OFFSET),
    HEURISTIC_BF       = 1ull << (HEURISTIC + CO_BF_OFFSET),
    RSVD_BF            = 1ull << (HEURISTIC + CO_BF_OFFSET),
    THORUP_SAMPLING_BF = 1ull << (THORUP_SAMPLING + AS_BF_OFFSET),
    D2_SAMPLING_BF = 1ull << (D2_SAMPLING + AS_BF_OFFSET),
    UNIFORM_SAMPLING_BF = 1ull << (UNIFORM_SAMPLING + AS_BF_OFFSET),
    GREEDY_SAMPLING_BF = 1ull << (GREEDY_SAMPLING + AS_BF_OFFSET),
    CS_BF_OFFSET = AS_BF_OFFSET + 4
    METRIC_KMEDIAN_BF = 1ull << (METRIC_KMEDIAN + CS_BF_OFFSET),
    EXPECTATION_MAXIMIZATION_BF = 1ull << (EXPECTATION_MAXIMIZATION + CS_BF_OFFSET),
    BLACK_BOX_BF = 1ull << (BLACK_BOX + CS_BF_OFFSET),
    GRADIENT_DESCENT_BF = 1ull << (GRADIENT_DESCENT + CS_BF_OFFSET),
    EXHAUSTIVE_SEARCH_BF = 1ull << (EXHAUSTIVE_SEARCH + CS_BF_OFFSET),
    OM_BF_OFFSET = CS_BF_OFFSET + 5
};

static_assert(OM_BF_OFFSET < 32, "must be < 32");


} // clustering

} // namespace minocore

#endif /* FGC_CLUSTERING_H__ */
