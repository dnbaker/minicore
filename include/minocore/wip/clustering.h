#ifndef FGC_CLUSTERING_H__
#define FGC_CLUSTERING_H__
#include "minocore/dist.h"
#include "minocore/util/exception.h"
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
#undef FROM_INT

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


template<typename Mat>
class LookupMatrixOracle {
    const Mat &mat_;
public:
    LookupMatrixOracle(const Mat &mat): mat_(mat) {}
    size_t size() const {return mat_.rows();}
    template<typename Sol>
    auto compute_distance(const Sol &x, size_t center_index, size_t point_index) const {
        assert(center_index < mat_.rows());
        assert(point_index < mat_.columns());
        return mat_(x[center_index], point_index);
    }
};

template<typename Mat>
auto make_lookup_data_oracle(const Mat &mat) {
    return LookupMatrixOracle<Mat>(mat);
}


template<typename DataOracle, typename MyClusteringTraits>
struct ClusteringSolverBase: public MyClusteringTraits {

    using centers_t     = typename MyClusteringTraits::centers_t;
    using costs_t       = typename MyClusteringTraits::costs_t;
    using assignments_t = typename MyClusteringTraits::assignments_t;
    using cost_t        = typename MyClusteringTraits::cost_t;
    using index_t       = typename MyClusteringTraits::index_t;
    using MyClusteringTraits::asn_method;
    using MyClusteringTraits::center_origin;
    using MyClusteringTraits::approx;
    using MyClusteringTraits::sampling_method;
    using MyClusteringTraits::opt;

    using FT = typename MyClusteringTraits::cost_t;
private:
    const DataOracle &data_oracle_;
    /*
     * DataOracle is the key for interfacing with the data.
     * It must provide:
     * 1. size() const method listing the number of points.
     * 2. compute_distance(const centers_t &centers, unsigned center_index, unsigned point_index)
     *
     *    For pre-computed matrices (e.g., metric distance matrix) with rows corresponding to centers,
     *    and columns corresponding to data points,
     *    DataOracle might have a mat_ field for the matrix and return
     *    `mat_(center_index, point_index)`.
     *    LookupMatrixOracle satisfies this, for instance.
     *
     *    For Applicator-supported functions, this might be
     *    `applicator_(point_index, centers_[center_index])`
     *    or have an alternate form that caches logs or sqrts.
     */
    size_t np_;
    size_t k_;

    std::unique_ptr<centers_t>     complete_sol_;
    std::unique_ptr<assignments_t> complete_assignments_;
    std::unique_ptr<costs_t>       complete_costs_;

public:
    void set_assignments_and_costs() {
        PREC_REQ(complete_sol_.get(), "Complete sol must already have been computed.");
        if constexpr(asn_method == HARD) {
            if(!complete_assignments_)
                complete_assignments_.reset(new assignments_t(data_oracle_.size()));
            else if(complete_assignments_->size() != data_oracle_.size())
                complete_assignments_->resize(data_oracle_.size());
            if(!complete_costs_)
                complete_costs_.reset(new costs_t(data_oracle_.size()));
            else if(complete_costs_->size() != data_oracle_.size())
                complete_costs_->resize(data_oracle_.size());
            OMP_PFOR
            for(size_t i = 0; i < data_oracle_.size(); ++i) {
                auto mincost = data_oracle_.compute_distance(*complete_sol_, 0, i);
                unsigned bestind = 0;
                for(size_t j = 1; j < complete_sol_->size(); ++j) {
                    if(auto newcost = data_oracle_.compute_distance(*complete_sol_, j, i); newcost < mincost)
                        mincost = newcost, bestind = j;
                }
                complete_assignments_->operator[](i) = bestind;
                complete_costs_->operator[](i) = mincost;
            }
        } else {
            assert(complete_sol_->size() == k_);
            if(!complete_costs_) {
                complete_costs_.reset(new costs_t(np_, k_));
            } else if(complete_costs_->rows() != np_ || complete_costs_->columns() != k_) {
                complete_costs_->resize(np_, k_);
            }
            if(!complete_assignments_) complete_assignments_.reset(new assignments_t(*complete_costs_));
            if(complete_assignments_->size() != data_oracle_.size())
                complete_assignments_->resize(data_oracle_.size());
            OMP_PFOR
            for(size_t i = 0; i < data_oracle_.size(); ++i) {
                auto cost_row = row(*complete_costs_, i, blaze::unchecked);
                cost_row[0] = data_oracle_.compute_distance(*complete_sol_, 0, i);
                for(size_t j = 1; j < complete_sol_->size(); ++j) {
                    cost_row[j] = data_oracle_.compute_distance(*complete_sol_, j, i);
                }
                if constexpr(MyClusteringTraits::asn_method == SOFT) {
                    cost_row /= blz::sum(cost_row);
                } else {
                    cost_row = blz::exp(cost_row - blz::max(cost_row));
                    cost_row /= blz::sum(cost_row);
                }
            }
        }
    }
    void set_centers(centers_t &&newcenters) {
        this->complete_sol_.reset(new centers_t(std::move(newcenters)));
    }
    ClusteringSolverBase(const DataOracle &data, size_t npoints, unsigned k): data_oracle_(data), np_(npoints), k_(k) {}
    double calculate_cost(const centers_t &centers) {
        throw NotImplementedError();
    }
    const assignments_t &get_assignments(bool recalc=true) {
        if(!complete_assignments_ || recalc) set_assignments_and_costs();
        return *complete_assignments_;
    }
};


} // clustering

} // namespace minocore

#endif /* FGC_CLUSTERING_H__ */
