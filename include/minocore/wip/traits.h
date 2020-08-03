#ifndef FGC_CLUSTERING_TRAITS_H__
#define FGC_CLUSTERING_TRAITS_H__

namespace minocore {
namespace clustering {

#pragma message("Note: clusteringtraits has been deprecated in favor of SumOpts")

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

static constexpr ce_t UNSET = ce_t(-1);

enum Assignment: ce_t {
    HARD = 0,
    /* Assignment(x) = argmin_{c \in C}[d(c, x)]
     *
     */
    SOFT = 1,
    // Assignment(X, c) = \frac{c}{\sum_{c' \in C}[d(c', x)]}
    SOFT_HARMONIC_MEAN = 2,
    /* Assignment(X, c) = \frac{e^{d(c, x)}}{sum_{c' \in C}[e^d(c', x)]}
     *                  = softmax(d(C, x))
     */
};
enum CenterOrigination: ce_t {
    INTRINSIC = 0,
    EXTRINSIC = 1
};

enum ApproximateSolutionType: ce_t {
    BICRITERIA      = 0,
    CONSTANT_FACTOR = 1,
    HEURISTIC       = 2,
    RSVD            = 3,
    DEFAULT_APPROX = UNSET
};
enum CenterSamplingType: ce_t {
    THORUP_SAMPLING,
    D2_SAMPLING,
    UNIFORM_SAMPLING,
    GREEDY_SAMPLING,
    DEFAULT_SAMPLING = UNSET,
    COST_SAMPLING = D2_SAMPLING,
};
enum OptimizationMethod: ce_t {
    METRIC_KMEDIAN,
    EXPECTATION_MAXIMIZATION,
    BLACK_BOX,
    GRADIENT_DESCENT,
    EXHAUSTIVE_SEARCH,
    DEFAULT_OPT = UNSET
};

enum MetricKMedianSolverMethod: ce_t {
    JAIN_VAZIRANI_FL,
    LOCAL_SEARCH,
    JV_PLUS_LOCAL_SEARCH,
    DEFAULT_SOLVER = UNSET
};



template<Assignment asn_method, typename index_t=uint32_t, typename cost_t=float>
using assignment_fmt_t = std::conditional_t<asn_method == HARD,
                                         blz::DV<index_t>,
                                         blaze::DynamicMatrix<cost_t>
                                        >;

template<typename FT=float, typename IT=uint32_t, Assignment asn=HARD, CenterOrigination co=EXTRINSIC>
struct ClusteringTraits {
    static constexpr Assignment asn_method = asn;
    static constexpr CenterOrigination center_origin = co;
    ApproximateSolutionType approx = static_cast<ApproximateSolutionType>(UNSET);
    CenterSamplingType sampling = static_cast<CenterSamplingType>(UNSET);
    OptimizationMethod opt = static_cast<OptimizationMethod>(UNSET);
    MetricKMedianSolverMethod metric_solver = JV_PLUS_LOCAL_SEARCH;

    static constexpr FT DEFAULT_EPS = 1e-6;

// Settings
    FT thorup_npermult = 7;
    FT approx_mul = 50;
    FT eps = DEFAULT_EPS;
    unsigned thorup_iter = 4;
    unsigned thorup_sub_iter = 10;
    unsigned max_jv_rounds = 100;
    unsigned max_lloyd_iter = 1000;
    unsigned k = -1;
    size_t npoints = 0;

    bool compute_full = true;
    uint64_t seed = 13;

    const FT *weights = nullptr;

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
                                       blz::DynamicMatrix<cost_t>>;
    // If hard assignment, then assignments are managed
    using assignments_t = assignment_fmt_t<asn_method, index_t, cost_t>;
    using centers_t = std::conditional_t<center_origin == INTRINSIC,
                                         blz::DV<index_t>,
                                         std::vector<blaze::DynamicVector<FT, blaze::rowVector>>
                                        >;
    // Thorup
};


template<typename FT, typename IT=uint32_t, Assignment asn_method=HARD, CenterOrigination co=INTRINSIC>
ClusteringTraits<FT, IT, asn_method, co> make_clustering_traits(
    size_t npoints, unsigned k,
    CenterSamplingType csample=DEFAULT_SAMPLING, OptimizationMethod opt=DEFAULT_OPT,
    ApproximateSolutionType approx=DEFAULT_APPROX, const FT *weights=nullptr, uint64_t seed=0,
    size_t max_iter=100, double eps=ClusteringTraits<FT, IT, asn_method, co>::DEFAULT_EPS) {
    ClusteringTraits<FT, IT, asn_method, co> ret;
    ret.k = k;
    ret.seed = seed;
    ret.max_jv_rounds = ret.max_lloyd_iter = max_iter;
    ret.eps = eps;
    ret.opt = opt;
    ret.sampling = csample;
    ret.approx = approx;
    ret.weights = weights;
    ret.npoints = npoints;
    return ret;
}


namespace detail {

template<typename MatrixType>
struct ApplicatorAdaptor {
    const jsd::DissimilarityApplicator<MatrixType> &mat_;
    ApplicatorAdaptor(const jsd::DissimilarityApplicator<MatrixType> &mat): mat_(mat) {}
    decltype(auto) operator()(size_t i, size_t j) const {
        return mat_(i, j);
    }
    auto get_measure() const {return mat_.get_measure();}
};
template<typename MatrixType>
auto make_aa(const jsd::DissimilarityApplicator<MatrixType> &mat) {
    return ApplicatorAdaptor<MatrixType>(mat);
}

} // namespace detail

} // clustering
} // minocore

#endif /* FGC_CLUSTERING_TRAITS_H__ */
