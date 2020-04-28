#ifndef FGC_CLUSTERING_TRAITS_H__
#define FGC_CLUSTERING_TRAITS_H__

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
    SOFTMAX = 2, //Currently unused
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
    ASN_BF_BITMASK = (1 << 3) - 1,
    HARD_BF = 1 << HARD,
    SOFT_BF = 1 << SOFT,
    SOFTMAX_BF = 1 << SOFTMAX,
    CO_BF_OFFSET = ASN_BF_OFFSET + 2,
    CO_BF_BITMASK = ((1 << (CO_BF_OFFSET - ASN_BF_OFFSET)) - 1) << ASN_BF_OFFSET,
    INTRINSIC_BF = 1 << ASN_BF_OFFSET,
    EXTRINSIC_BF = 1 << (EXTRINSIC + ASN_BF_OFFSET),
    AS_BF_OFFSET = CO_BF_OFFSET + 4,
    AS_BF_BITMASK = ((1 << (AS_BF_OFFSET - CO_BF_OFFSET)) - 1) << CO_BF_OFFSET,
    BICRITERIA_BF      = 1ull << (BICRITERIA + CO_BF_OFFSET),
    CONSTANT_FACTOR_BF = 1ull << (CONSTANT_FACTOR + CO_BF_OFFSET),
    HEURISTIC_BF       = 1ull << (HEURISTIC + CO_BF_OFFSET),
    RSVD_BF            = 1ull << (HEURISTIC + CO_BF_OFFSET),
    THORUP_SAMPLING_BF = 1ull << (THORUP_SAMPLING + AS_BF_OFFSET),
    D2_SAMPLING_BF = 1ull << (D2_SAMPLING + AS_BF_OFFSET),
    UNIFORM_SAMPLING_BF = 1ull << (UNIFORM_SAMPLING + AS_BF_OFFSET),
    GREEDY_SAMPLING_BF = 1ull << (GREEDY_SAMPLING + AS_BF_OFFSET),
    CS_BF_OFFSET = AS_BF_OFFSET + 4,
    CS_BF_BITMASK = ((1 << (CS_BF_OFFSET - AS_BF_OFFSET)) - 1) << AS_BF_OFFSET,
    METRIC_KMEDIAN_BF = 1ull << (METRIC_KMEDIAN + CS_BF_OFFSET),
    EXPECTATION_MAXIMIZATION_BF = 1ull << (EXPECTATION_MAXIMIZATION + CS_BF_OFFSET),
    BLACK_BOX_BF = 1ull << (BLACK_BOX + CS_BF_OFFSET),
    GRADIENT_DESCENT_BF = 1ull << (GRADIENT_DESCENT + CS_BF_OFFSET),
    EXHAUSTIVE_SEARCH_BF = 1ull << (EXHAUSTIVE_SEARCH + CS_BF_OFFSET),
    OM_BF_OFFSET = CS_BF_OFFSET + 5,
    OM_BF_BITMASK = ((1 << ((OM_BF_OFFSET - CS_BF_OFFSET))) - 1) << CS_BF_OFFSET
};
static constexpr ce_t to_bitfield(Assignment asn) {
    return ce_t(1) << asn;
}
static constexpr ce_t to_bitfield(CenterOrigination co) {
    return ce_t(1) << (co + ASN_BF_OFFSET);
}
static constexpr ce_t to_bitfield(ApproximateSolutionType as) {
    return ce_t(1) << (as + CO_BF_OFFSET);
}
static constexpr ce_t to_bitfield(CenterSamplingType sampling) {
    return ce_t(1) << (sampling + AS_BF_OFFSET);
}
static constexpr ce_t to_bitfield(OptimizationMethod opt) {
    return ce_t(1) << (opt + CS_BF_OFFSET);
}

static constexpr ce_t
    to_bitfield(Assignment asn, CenterOrigination co, ApproximateSolutionType as,
                CenterSamplingType sampling, OptimizationMethod opt) {
    return to_bitfield(asn) | to_bitfield(co) | to_bitfield(as) | to_bitfield(sampling) | to_bitfield(opt);
}

namespace detail {
constexpr size_t cl2(const size_t n) {
    switch(n) {
        case 0: return size_t(-1);
        case 1: return 0;
        default: return 1 + cl2(n / 2);
    }
}

}
template<typename T>
constexpr T from_integer(ce_t v) {
    throw std::runtime_error("Illegal type T");
}
template<> constexpr Assignment from_integer(ce_t v) {
    return static_cast<Assignment>(detail::cl2((v & ASN_BF_BITMASK) >> 0));
}
template<> constexpr CenterOrigination from_integer(ce_t v) {
    return static_cast<CenterOrigination>(detail::cl2((v & CO_BF_BITMASK) >> ASN_BF_OFFSET));
}
template<> constexpr ApproximateSolutionType from_integer(ce_t v) {
    return static_cast<ApproximateSolutionType>(detail::cl2((v & AS_BF_BITMASK) >> CO_BF_OFFSET));
}
template<> constexpr CenterSamplingType from_integer(ce_t v) {
    return static_cast<CenterSamplingType>(detail::cl2((v & CS_BF_BITMASK) >> AS_BF_OFFSET));
}
template<> constexpr OptimizationMethod from_integer(ce_t v) {
    return static_cast<OptimizationMethod>(detail::cl2((v & OM_BF_BITMASK) >> CS_BF_OFFSET));
}

static_assert(OM_BF_OFFSET < 32, "must be < 32");

template<typename FT, typename IT, ce_t bf>
struct ClusteringTraits {
    static constexpr Assignment asn_method = from_integer<Assignment>(bf);
    static constexpr CenterOrigination center_origin =
        from_integer<CenterOrigination>(bf);
    static constexpr ApproximateSolutionType approx =
        from_integer<ApproximateSolutionType>(bf);
    static constexpr CenterSamplingType sampling_method = from_integer<CenterSamplingType>(bf);
    static constexpr OptimizationMethod opt = from_integer<OptimizationMethod>(bf);
    static_assert(std::is_floating_point_v<FT>, "FT must be floating");
    static_assert(std::is_integral_v<IT>, "FT must be integral and support required index ranges");
    using cost_t = FT;
    using index_t = IT;

    // If hard, one cost per point
    // If soft, one cost per point per center
    // Assignment fractions are generated as-needed (for the case of softmax)
    // For this reason, matrix forms are stored as
    // row = point, column = center
    using costs_t = std::conditional_t<asn_method == HARD,
                                       blz::DV<cost_t>,
                                       blaze::DynamicMatrix<cost_t>>;
    // If hard assignment, then assignments are managed
    using assignments_t = std::conditional_t<asn_method == HARD,
                                             blz::DV<index_t>,
                                             blaze::DynamicMatrix<cost_t>
                                            >;
    using centers_t = std::conditional_t<center_origin == INTRINSIC,
                                         blz::DV<index_t>,
                                         std::vector<blaze::DynamicVector<FT, blaze::rowVector>>
                                        >;
};

template<typename T>
struct is_clustering_traits: public std::false_type {};

template<typename FT, typename IT, ce_t bf>
struct is_clustering_traits<ClusteringTraits<FT, IT, bf>>: public std::true_type {};

template<typename T>
static constexpr bool is_clustering_traits_v = is_clustering_traits<T>::value;


template<typename FT=float, typename IT=uint32_t, Assignment asn=HARD, CenterOrigination co=EXTRINSIC, ApproximateSolutionType approx=CONSTANT_FACTOR,
         CenterSamplingType sampling=THORUP_SAMPLING, OptimizationMethod opt=METRIC_KMEDIAN>
struct Meta: public ClusteringTraits<FT, IT, to_bitfield(asn, co, approx, sampling, opt)> {};

} // clustering
} // minocore

#endif /* FGC_CLUSTERING_TRAITS_H__ */
