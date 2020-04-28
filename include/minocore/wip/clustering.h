#ifndef FGC_CLUSTERING_H__
#define FGC_CLUSTERING_H__
#include "minocore/dist.h"
#include "minocore/util/exception.h"
#include "minocore/wip/clustering_traits.h"
#include <cstdint>

namespace minocore {

namespace clustering {


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
    uint32_t k_, points_to_sample_;

    std::unique_ptr<centers_t>     c_sol_;
    std::unique_ptr<assignments_t> c_assignments_;
    std::unique_ptr<costs_t>       c_costs_;
    std::unique_ptr<cost_t>        pointwise_costs_;
    const FT *weights_;

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
                asn_row = blz::exp(cost_row - blz::max(cost_row));
                asn_row /= blz::sum(asn_row);
            }
        }
    }
    void approx_sol(uint64_t seed=0) {
        if constexpr(opt == BLACK_BOX || opt == GRADIENT_DESCENT || opt == EXHAUSTIVE_SEARCH)
            throw NotImplementedError("Optimization under black box, gd or exhaustive search not yet supported");
        if constexpr(asn_method != HARD) 
            throw NotImplementedError("Not completed yet: SOFT or SOFTMAX clustering");
        else
        {
            // One optimization technique each for metric (JV + local search)
            //                                 and expectation maximization.
            if constexpr(opt == EXPECTATION_MAXIMIZATION) {
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
                static_assert(opt == METRIC_KMEDIAN, "annotate that this is metric kmedian");
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
            sampler.make_sampler(np_, points_to_sample_, ptr, c_assignments_->data(), weights_, seed);
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
                         unsigned points_to_sample=0, const FT *weights=nullptr):
        data_oracle_(data), np_(npoints), k_(k),
        points_to_sample_(points_to_sample ? points_to_sample: k_),
        weights_(weights)
    {
        if(points_to_sample_ != k_) std::fprintf(stderr, "note: sampling different number of points");
    }
    double calculate_cost(const centers_t &centers) {
        throw NotImplementedError();
    }
    const assignments_t &get_assignments(bool recalc=true) {
        if(!c_assignments_ || recalc) set_assignments_and_costs();
        return *c_assignments_;
    }
};


} // clustering

} // namespace minocore

#endif /* FGC_CLUSTERING_H__ */
