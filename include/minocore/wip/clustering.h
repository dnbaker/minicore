#ifndef FGC_CLUSTERING_H__
#define FGC_CLUSTERING_H__
#include "minocore/dist.h"
#include "minocore/util/exception.h"
#include "minocore/wip/clustering_traits.h"
#include <cstdint>

namespace minocore {

namespace clustering {


struct AveragePolicy {
    template<typename VT, bool TF, typename Range, typename VT2>
    void perform_average(blz::DenseVector<VT, TF> &ret, const Range &r, const blz::Vector<VT2, TF> *wc = static_cast<const blz::Vector<VT2, TF> *>(nullptr)) {
        if(wc) {
            assert((~(*wc)).size() == r.rows());
            assert(blz::expand(~(*wc), r.columns()).rows() == r.rows());
            assert(blz::expand(~(*wc), r.columns()).columns() == r.columns());
            ~ret = blz::mean<blz::columnwise>(r % blz::expand(~(*wc), r.columns())
        } else {
            ~ret = blz::mean<blz::columnwise>(r);
        }
    }
};

struct MedianPolicy {
    template<typename VT, bool TF, typename Range, typename WeightContainer=const blz::ElementType_t<VT> *>
    void perform_average(blz::DenseVector<VT, TF> &ret, const Range &r, const blz::Vector<VT2, TF> *wc = static_cast<const blz::Vector<VT2, TF> *>(nullptr)) {
        coresets::l1_median(r, ret,  wc);
    }
};

template<typename MatrixType, typename WFT=blz::ElementType_t<MatrixType> >
void perform_cluster_metric_kmedian(const jsd::DissimilarityApplicator<MatrixType> &app, unsigned k, uint64_t seed=0, const WFT *weights=static_cast<WFT *>(nullptr))
{
    throw NotImplementedError();
}

enum LloydLoopResult {
    OK,
    REACHED_MAX_ROUNDS,
    UNFINISHED
};
template<Assignment asn_method=HARD, CenterOrigination co=EXTRINSIC, typename MatrixType, typename CentersType, typename CentroidPolicy, typename Assignments>
LloydLoopResult perform_lloyd_loop(CentersType &centers, Assignments &assignments, &const jsd::DissimilarityApplicator<MatrixType> &app, unsigned k, uint64_t seed=0, const WFT *weights=static_cast<WFT *>(nullptr),
                        CentroidPolicy centroidpol, size_t max_iter=100, double eps=1e-4, LloydLoopResult &ret)
{
    if(co != EXTRINSIC) throw std::invalid_argument("Must be extrinsic for Lloyd's");
    using FT = ElementType_t<MatrixType>;
    auto &mat = app.data();
    CentersType centers_cpy(centers);
    double last_distance = std::numeric_limits<double>::max(), first_distance = last_distance,
           center_distance;
    LloydLoopResult ret = UNFINISHED;
    auto get_center_change_distance = [&]() {
        center_distance = std::accumulate(centers_cpy.begin(), centers_cpy.end(), 0.,
            [&](double value, auto &center) {
                auto ind = std::distance(*centers_cpy.begin(), &centers_);
                return value + blz::sum(blz::abs(center - centers_[ind]));
            }
        );
        std::swap(centers_cpy, centers);
        if(last_distance == std::numeric_limits<double>::max()) {
            last_distance = first_distance = center_distance;
            iternum = 1;
        } else {
            last_distance = center_distance;
            if(center_distance / first_distance < eps)
                ret = LloydLoopResult::OK;
            if(++iternum > max_iter)
                ret = LloydLoopResult::REACHED_MAX_ROUNDS;
                ret = UNFINISHED;
        }
    };
    // Next: make a set of std::vectors, then use blaze to compute averages under the policy
    // Everything but L1 and TVD use element-wise mean
    if constexpr(asn_method = HARD) {
        std::vector<std::vector<uint32_t>> assigned(centers.size());
        OMP_ONLY(std::unique_ptr<std::mutex[]> mutexes(centers.size());)
        size_t iternum = 0;
        for(;;) {
            // Do it forever
            for(auto &i: assigned) i.clear();
            OMP_PFOR
            for(size_t i = 0; i < app.size(); ++i) {
                auto dist = app(i, centers_cpy[0]);
                unsigned asn = 0;
                for(size_t j = 1; j < centers_cpy.size(); ++j) {
                    auto newdist = app(i, centers_cpy[j]);
                    if(newdist < dist) {
                        asn = j;
                        dist = newdist;
                    }
                }
                assignments[i] = asn;
                {
                    OMP_ONLY(std::unique_lock<std::mutex> lock(mutexes[asn]);)
                    assigned[asn].push_back(i);
                }
            }
            // Make assignments
            for(size_t i = 0; i < centers_cpy.size(); ++i) {
                auto &cref = centers_cpy[i];
                auto &assigned_ids = assigned[i];
                shared::sort(assigned_ids.begin(), assigned_ids.end()); // Better access pattern
                centroidpol.perform_average(cref, rows(mat, assigned_ids.data(), assigned_ids.size()), weights);
            }
            get_center_change_distance();
            if(ret != UNFINISHED) return ret;
        }
        // Set the returned values to be the last iteration's.
    } else {
        size_t iternum = 0;
        const size_t nc = centers.size(), nr = app.size();
        if(assignments.rows() != app.size() || assignments.columns() != centers.size()) {
            assignments.resize(app.size(), centers.size());
        }
        for(;;) {
            for(auto &c: centers) c = static_cast<FT>(0);
            OMP_PFOR
            for(size_t i = 0; i < nr; ++i) {
                blz::DV<FT, blz::rowVector> tmp(centers.size());
                auto row = row(assignments, i, BLAZE_CHECK_DEBUG);
                for(unsigned j = 0; j < nc; ++j) {
                    row[j] = app(i, centers_cpy[j]);
                }
                if constexpr(asn_method == SOFT_HARMONIC_MEAN)
                    tmp = 1. / row;
                else 
                    tmp = blz::exp(-row + blz::min(row));
                tmp = tmp * (1. / blz::sum(row));
                // And then compute its contribution to the mean of the points.
                // Use stable running mean calculation
                throw NotImplementedError("Finish this");
            }
        }
        get_center_change_distance();
        if(ret != UNFINISHED) return ret;
        throw NotImplementedError("Not yet finished");
    }
    // Then, iteratively perform procedure
}


template<Assignment asn_method=HARD, CenterOrigination co=INTRINSIC, typename MatrixType, typename IT=uint32_t>
auto perform_clustering(const jsd::DissimilarityApplicator<MatrixType> &app, unsigned k, CenterSamplingType csample=DEFAULT_SAMPLING,
                        const blz::ElementType_t<MatrixType> *weights=nullptr, uint64_t seed=0, OptimizationMethod opt=DEFAULT_OPT) {
    using FT = typename MatrixType::ElementType;
    ClusteringTraits<FT, IT, asn_method, co> clustering_traits;
    clustering_traits.sampling = csample;
    typename ClusteringTraits<FT, IT, asn_method, co>::centers_t centers;
    typename ClusteringTraits<FT, IT, asn_method, co>::assignments_t assigments;
    auto measure = app.measure_;
    if(opt == DEFAULT_OPT) {
        switch(measure) {
            case L2:
            case SQRL2:
            case L1: case TVD:
            case COSINE_DISTANCE:
            case PROBABILITY_COSINE_DISTANCE:
            case LLR: case UWLLR:
            case HELLINGER: case BHATTACHARYYA_DISTANCE:
                opt = EXPECTATION_MAXIMIZATION; break;
            /*
             * Bregman Divergences, LLR, cosine distance use the (weighted) mean of each
             * point, in either soft or hard clustering.
             * TVD and L1 use the feature-wise median.
             * Scores are either calculated with softmax distance or harmonic softmax
             */
            case ORACLE_METRIC: case ORACLE_PSEUDOMETRIC: case BHATTACHARYYA_METRIC: case WASSERSTEIN:
                /* otherwise, use metric kmedian */
                opt = METRIC_KMEDIAN; break;
            default:
                if(blz::detail::is_bregman(opt)) {
                    opt = EXPECTATION_MAXIMIZATION;
                    break;
                }
        }
    }


    if(blz::detail::satisfies_d2(measure) || measure == blz::L1 || measure == blz::TOTAL_VARIATION_DISTANCE) {
        auto [initcenters, initasn, initcosts] = jsd::make_kmeanspp(app, k, seed, weights);
         
        if constexpr(co == INTRINSIC) {
            throw std::invalid_argument("Shouldn't happen");
        }
        centers.reserve(k);
        std::copy(initasn.begin(), initasn.end(), std::back_inserter(assignments));
        for(const auto id: initcenters) {
            centers.emplace_back(row(app.data(), id));
        }
        if(measure == blz::L1 || measure == blz::TOTAL_VARIATION_DISTANCE) {
            if(co == INTRINSIC || opt == METRIC_KMEDIAN) {
                // Do graph metric calculation
                perform_cluster_metric_kmedian(app, k, seed, weights);
            } else {
                // Do Lloyd's loop (``kmeans'' algorithm)
                perform_lloyd_loop<asn_method>(centers, assignments, app, k, seed, weights);
            }
        } else  {
            if constexpr(co == EXTRINSIC) {
                    // Do Lloyd's loop with hard assignment
            } else {
                // Do Lloyd's loop with soft assignment
            }
        }
    } else if(measure == blz::L1) {
        auto [initcenters, initasn, initcosts] = jsd::make_kmeanspp(app, k, seed, weights);
        // Use kmeans++ style initialiation
    } else if(blz::detail::is_symmetric(measure)) {
        throw std::runtime_error("Not implemented: symmetric measure clustering. This method should perform sampling (governed by the csample variable)"
                                  ", followed by facility location, and finished by local search.");
        dm::DistanceMatrix<FT, 0, dm::DM_MMAP> distmat(app.size());
        app.set_distance_matrix(distmat);
    } else {
        throw NotImplementedError("Unsupported: asymmetric measures not supporting D2 sampling");
    }
}

#if 0
namespace helpers {

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
    void operator[](size_t ) const {
        throw std::runtime_error("This should never be called");
    }
    auto compute_distance(nullptr_t, size_t center_index, size_t point_index) const {
        assert(point_index < mat_.rows());
        assert(center_index < mat_.rows());
        return mat_(center_index, point_index);
    }
};

template<typename Mat>
auto make_lookup_data_oracle(const Mat &mat) {
    return LookupMatrixOracle<Mat>(mat);
}

template<typename Mat, typename Functor>
class ExtrinsicFunctorOracle {
    const Mat &mat_;
    const Functor &func_;
public:
    ExtrinsicFunctorOracle(const Mat &mat, const Functor &func): mat_(mat), func_(func) {}
    size_t size() const {return mat_.rows();}
    template<typename Sol>
    auto compute_distance(const Sol &x, size_t center_index, size_t point_index) const {
        assert(point_index < mat_.rows());
        assert(center_index < x.size());
        return func_(x[center_index], mat_[point_index]);
    }
    decltype(auto) operator[](size_t ind) {return mat_[ind];}
    decltype(auto) operator[](size_t ind) const {return mat_[ind];}
    // This function computes a distance between two points
    auto compute_distance(nullptr_t, size_t center_index, size_t point_index) const {
        return compute_distance(center_index, point_index);
    }
    auto compute_distance(size_t center_index, size_t point_index) const {
        assert(point_index < size());
        assert(center_index < size());
        return func_(mat_[center_index], mat_[point_index]);
    }
};

template<typename Mat, typename Func>
auto make_exfunc_oracle(const Mat &mat, const Func &func) {
    return ExtrinsicFunctorOracle<Mat, Func>(mat, func);
}

} // helpers
using helpers::make_exfunc_oracle;
using helpers::make_lookup_data_oracle;


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
    //using MyClusteringTraits::sampling_method;
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
     *    For distance-oracle functions,
     *    use the ExtrinsicFunctorOracle class.
     *
     *    For instance, if `dm` is a dense matrix of points in row-major format:
     *    auto oracle = clustering::make_exfunc_oracle(dm, blz::sqrL2Norm())
     *    clustering::ClusteringSolverBase<decltype(oracle), MatrixMeta> solver(oracle, dm.rows(), k);
     *
     *
     *    For Applicator-supported functions, this might be
     *    `applicator_(point_index, centers_[center_index])`
     *    or have an alternate form that caches logs or sqrts.
     */
    size_t np_;
    uint32_t k_;
    uint32_t points_to_sample_;
    DissimilarityMeasure measure_; // What measure of dissimilarity.
                                   // Use ORACLE_METRIC or ORACLE_PSEUDOMETRIC as placeholders for measures
                                   // Not supported by the applicator

    std::unique_ptr<centers_t>     c_sol_;
    std::unique_ptr<assignments_t> c_assignments_;
    std::unique_ptr<costs_t>       c_costs_;
    std::unique_ptr<cost_t>        pointwise_costs_;
    const FT *weights_;
    SensitivityMethod sens_; // Which coreset construction method

    void validate_parameters() {
        assert(sens_ != static_cast<SensitivityMethod>(-1));
        if(opt == METRIC_KMEDIAN) {
            validate(blz::detail::satisfies_metric(measure_) || blz::detail::satisfies_rho_metric(measure_));
        }
#if 0
        if(sampling_method == THORUP_SAMPLING) {
            validate(blz::detail::satisfies_metric(measure_) || blz::detail::satisfies_rho_metric(measure_));
        }
        if(sampling_method == D2_SAMPLING) {
            validate(blz::detail::satisfies_d2(measure_));
        }
#endif
    }

    void set_sensitivity_method(SensitivityMethod val=static_cast<SensitivityMethod>(-1)) {
        bool unset = val == static_cast<SensitivityMethod>(-1);
        if(unset) {
            if(blz::detail::is_bregman(val)) sens_ = LBK;
            else if(approx == BICRITERIA) {
                sens_ = BFL;
            } else if(approx == CONSTANT_FACTOR) {
                if(blz::detail::is_bregman(val)) std::fprintf(stderr, "Warning: Bregman approximations are O(log(k)) approximate, not constant.\n");
                sens_ = VX;
            } else /*approx == HEURISTIC */ {
                sens_ = BFL;
            }
        } else {
            if(val == VX) {
                MINOCORE_VALIDATE(approx == CONSTANT_FACTOR || approx == HEURISTIC);
                if(blz::detail::is_bregman(val)) std::fprintf(stderr, "Warning: Bregman solutions are O(log(k)) approximate, not constant.\n");
            } else if (val == LUCIC_FAULKNER_KRAUSE_FELDMAN) {
                throw NotImplementedError("Not supported currently: GMM coreset sampling");
            }
            sens_ = val;
        }
    }

public:
    void set_assignments_and_costs() {
        PREC_REQ(c_sol_.get(), "Complete sol must already have been computed.");
        if constexpr(asn_method == HARD) {
            if(!c_assignments_)
                c_assignments_.reset(new assignments_t(data_oracle_.size()));
            else if(c_assignments_->size() != data_oracle_.size())
                c_assignments_->resize(data_oracle_.size());
            if(!c_costs_)
                c_costs_.reset(new costs_t(data_oracle_.size()));
            else if(c_costs_->size() != data_oracle_.size())
                c_costs_->resize(data_oracle_.size());
            OMP_PFOR
            for(size_t i = 0; i < data_oracle_.size(); ++i) {
                auto mincost = data_oracle_.compute_distance(*c_sol_, 0, i);
                unsigned bestind = 0;
                for(size_t j = 1; j < c_sol_->size(); ++j) {
                    if(auto newcost = data_oracle_.compute_distance(*c_sol_, j, i); newcost < mincost)
                        mincost = newcost, bestind = j;
                }
                c_assignments_->operator[](i) = bestind;
                c_costs_->operator[](i) = mincost;
            }
        } else { // Soft or softmax assignments
            assert(c_sol_->size() == k_);
            if(!c_costs_) {
                c_costs_.reset(new costs_t(np_, k_));
            } else if(c_costs_->rows() != np_ || c_costs_->columns() != k_) {
                c_costs_->resize(np_, k_);
            }
            if(!c_assignments_) c_assignments_.reset(new assignments_t(*c_costs_));
            if(c_assignments_->size() != data_oracle_.size())
                c_assignments_->resize(data_oracle_.size());
            OMP_PFOR
            for(size_t i = 0; i < data_oracle_.size(); ++i) {
                // Compute costs
                auto cost_row = row(*c_costs_, i, blaze::unchecked);
                cost_row[0] = data_oracle_.compute_distance(*c_sol_, 0, i);
                for(size_t j = 1; j < c_sol_->size(); ++j) {
                    cost_row[j] = data_oracle_.compute_distance(*c_sol_, j, i);
                }
                // Use costs to make fractional assignments
                auto asn_row = row(*c_assignments_, i, blaze::unchecked);
                if(asn_method == SOFT)
                    asn_row = blz::exp(-cost_row + blz::min(cost_row));
                else // SOFT_HARMONIC_MEAN, actually harmonic mean
                    asn_row = 1. / cost_row;
                asn_row /= blz::sum(asn_row);
            }
        }
    }
    void approx_sol(uint64_t seed=0) {
        if constexpr(opt == BLACK_BOX || opt == GRADIENT_DESCENT || opt == EXHAUSTIVE_SEARCH)
            throw NotImplementedError("Optimization under black box, gd or exhaustive search not yet supported");
        if constexpr(asn_method != HARD)
            throw NotImplementedError("Not completed yet: SOFT or SOFT_HARMONIC_MEAN clustering");
        else
        {
            // One optimization technique each for metric (JV + local search)
            //                                 and expectation maximization.
            if(blz::detail::satisfies_d2(measure_)) {
                auto func = [&](size_t i, size_t j) {
                    return data_oracle_.compute_distance(i, j);
                };
                wy::WyRand<uint64_t, 2> rng(seed);
                auto [initcenters, initasn, initcosts] = coresets::kmeanspp(func, rng, np_, k_);
                std::vector<blz::DV<cost_t, blz::rowVector>> centers;
                centers.reserve(k_);
                for(const auto id: initcenters) {
                    centers.emplace_back(data_oracle_[id]);
                }
                set_centers(std::move(centers));
                set_assignments_and_costs();
            } else {
                throw NotImplementedError("Metric K-median needs to have optimizers plugged in.");
            }
        }
    }
    auto make_coreset_sampler(uint64_t seed=0) {
        PREC_REQ(this->c_costs_.get(), "Current costs must be calculated");
        const cost_t *ptr;
        if constexpr(asn_method == HARD) {
            // Use the c_costs->data() method.
            if(!weights_)
                ptr = c_costs_->data();
            else if(pointwise_costs_.get()) ptr = pointwise_costs_.get();
            else {
                pointwise_costs_.reset(new cost_t[np_]);
                blaze::CustomVector<FT, blaze::unaligned, blaze::unpadded>
                    pv(c_costs_->data(), np_), pc(pointwise_costs_.get(), np_);
                const blaze::CustomVector<FT, blaze::unaligned, blaze::unpadded> wv(const_cast<FT *>(weights_), np_);
                pc = pv * wv;
                ptr = pointwise_costs_.get();
            }
        } else {
            if(pointwise_costs_.get()) {
                ptr = pointwise_costs_.get();
            } else {
                pointwise_costs_.reset(new cost_t[np_]);
                OMP_PFOR
                for(size_t i = 0; i < np_; ++i)
                    pointwise_costs_[i] = blz::dot(row(*c_assignments_, i, blz::unchecked),
                                                   row(*c_costs_, i, blz::unchecked)) * getw(i);
                if(weights_) {
                    blaze::CustomVector<FT, blaze::unaligned, blaze::unpadded>
                        pv(pointwise_costs_.get(), np_);
                    const blaze::CustomVector<FT, blaze::unaligned, blaze::unpadded> wv(const_cast<FT *>(weights_), np_);
                    pv *= wv;
                }
            }
        }
        coresets::CoresetSampler<cost_t, index_t> sampler;
        if constexpr(asn_method == HARD) throw NotImplementedError("Coreset sampler supporting fractional assignment not yet available.");
        else {
            sampler.make_sampler(np_, points_to_sample_, ptr, c_assignments_->data(), weights_, seed, sens_);
        }
    }
    template<typename OT>
    void set_centers(const OT &centers) {
        this->c_sol_.reset(new centers_t(centers.size()));
        std::copy(centers.begin(), centers.end(), this->c_sol_->begin());
    }
    void set_centers(centers_t &&newcenters) {
        this->c_sol_.reset(new centers_t(std::move(newcenters)));
    }
    ClusteringSolverBase(const DataOracle &data, size_t npoints, unsigned k,
                         DissimilarityMeasure measure=ORACLE_PSEUDOMETRIC,
                         blz::distance::SensitivityMethod sens=static_cast<blz::distance::SensitivityMethod>(-1),
                         unsigned points_to_sample=0, const FT *weights=nullptr):
        data_oracle_(data), np_(npoints), k_(k),
        points_to_sample_(points_to_sample ? points_to_sample: k_),
        measure_(measure),
        weights_(weights)
    {
        if(points_to_sample_ != k_) std::fprintf(stderr, "note: sampling different number of points");
        set_sensitivity_method(sens);
        validate_parameters();
    }
    double calculate_cost(const centers_t &centers) {
        throw NotImplementedError();
    }
    const assignments_t &get_assignments(bool recalc=true) {
        if(!c_assignments_ || recalc) set_assignments_and_costs();
        return *c_assignments_;
    }
};
#endif



} // clustering

} // namespace minocore

#endif /* FGC_CLUSTERING_H__ */
