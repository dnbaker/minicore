#ifndef MINOCORE_CLUSTERING_CENTROID_H__
#define MINOCORE_CLUSTERING_CENTROID_H__
#include "minicore/util/blaze_adaptor.h"
#include "minicore/util/csc.h"
#include "minicore/dist.h"
#include "minicore/optim/kmedian.h"

namespace minicore { namespace clustering {

using blaze::unchecked;

enum CentroidPol {
    FULL_WEIGHTED_MEAN, // SQRL2, Bregman Divergences (+ convex combinations), cosine distance
    L1_MEDIAN,          // L1
    TVD_MEDIAN,         // Total variation distance, which is L1 in probability space
    GEO_MEDIAN,         // L2 norm
    JSM_MEDIAN,         // Unknown as of yet, but we will try weighted mean for now
    NOT_APPLICABLE
};

static constexpr const char *cp2str(CentroidPol pol) {
    switch(pol) {
     case FULL_WEIGHTED_MEAN: return "full weighted mean";
     case L1_MEDIAN:          return "l1 median";
     case TVD_MEDIAN:         return "tvd median";
     case GEO_MEDIAN:         return "geo median";
     case JSM_MEDIAN:         return "jsm median, same as full for now";
     default:
     case NOT_APPLICABLE:     return "not applicable";
    }
}


template<typename CtrT, typename VT, bool TF>
void set_center(CtrT &lhs, const blaze::Vector<VT, TF> &rhs) {
    if(lhs.size() != (*rhs).size()) {
        lhs.resize((*rhs).size());
    }
    if constexpr(blaze::IsSparseVector_v<CtrT>) {
        lhs.reserve((*rhs).size());
    }
    lhs = *rhs;
}

template<typename CtrT, typename VT, typename IT>
void set_center(CtrT &lhs, const util::CSparseVector<VT, IT> &rhs) {
    lhs.reserve(rhs.nnz());
    if(lhs.size() != rhs.dim_) lhs.resize(rhs.dim_);
    lhs.reset();
    for(const auto &pair: rhs) lhs[pair.index()] = pair.value();
}

template<typename CtrT, typename DataT, typename IndicesT, typename IndPtrT, typename IT, typename WeightT=blz::DV<blz::ElementType_t<DataT>>>
void set_center(CtrT &ctr, const util::CSparseMatrix<DataT, IndicesT, IndPtrT> &mat, IT *asn, size_t nasn, WeightT *w = static_cast<WeightT>(nullptr))
{
    using VT = std::conditional_t<std::is_floating_point_v<DataT>, DataT, std::conditional_t<(sizeof(DataT) < 8), float, double>>;
    blz::DV<VT, blz::TransposeFlag_v<CtrT>> mv(mat.columns(), VT(0));
    double wsum = 0.;
    OMP_PFOR_DYN
    for(size_t i = 0; i < nasn; ++i) {
        for(const auto &pair: row(mat, asn[i])) {
            VT v = pair.value();
            if(w) {
                auto wv = (*w)[asn[i]];
                OMP_ATOMIC
                wsum += wv;
                v *= wv;
            }
            OMP_ATOMIC
            mv[pair.index()] += v;
        }
    }
    if(!wsum) wsum = nasn;
    ctr = mv / wsum;
}

template<typename CtrT, typename DataT, typename IndicesT, typename IndPtrT, typename IT, typename WeightT>
void set_center_l2(CtrT &center, const util::CSparseMatrix<DataT, IndicesT, IndPtrT> &mat, IT *asp, size_t nasn, WeightT *weights, double eps=0.) {
    util::geomedian(mat, center, asp, nasn, weights, eps);
}


template<typename VT, typename Alloc, typename IT>
decltype(auto) elements(const std::vector<VT, Alloc> &w, IT *asp, size_t nasn) {
    return elements(blz::make_cv(&w[0], w.size()), asp, nasn);
}

template<typename CtrT, typename MT, typename IT, typename WeightT>
void set_center_l2(CtrT &center, const blaze::Matrix<MT, blaze::rowMajor> &mat, IT *asp, size_t nasn, WeightT *weights, double eps=0.) {
    auto rowsel = rows(mat, asp, nasn);
    VERBOSE_ONLY(std::cerr << "Calculating geometric median for " << nasn << " rows and storing in " << center << '\n';)
    if(weights)
        blz::geomedian(rowsel, center, elements(*weights, asp, nasn), eps);
    else
        blz::geomedian(rowsel, center, eps);
    VERBOSE_ONLY(std::cerr << "Calculated geometric median; new values: " << ctrs[i] << '\n';)
}


template<typename CtrT, typename MT, bool SO, typename IT, typename WeightT=blz::DV<blz::ElementType_t<MT>>>
void set_center(CtrT &ctr, const blaze::Matrix<MT, SO> &mat, IT *asp, size_t nasn, WeightT *w = static_cast<WeightT*>(nullptr)) {
    auto rowsel = rows(*mat, asp, nasn);
    if(w) {
        auto elsel = elements(*w, asp, nasn);
        auto weighted_rows = rowsel % blaze::expand(elsel, (*mat).columns());
        // weighted sum over total weight -> weighted mean
        ctr = blaze::sum<blaze::columnwise>(weighted_rows) / blaze::sum(elsel);
    } else {
        ctr = blaze::mean<blaze::columnwise>(rowsel);
    }
}


using namespace ::minicore::distance;

static constexpr INLINE CentroidPol msr2pol(distance::DissimilarityMeasure msr) {
    switch(msr) {
        case EMD: case WEMD:
        case ORACLE_METRIC: case ORACLE_PSEUDOMETRIC:
        default:
            return NOT_APPLICABLE;


        case UWLLR: case LLR: case MKL: case JSD: case SQRL2: case POISSON:
        case REVERSE_POISSON: case REVERSE_MKL: case ITAKURA_SAITO: case REVERSE_ITAKURA_SAITO:
        case SYMMETRIC_ITAKURA_SAITO: case RSYMMETRIC_ITAKURA_SAITO:

        case SRULRT: case SRLRT: case JSM:
            return JSM_MEDIAN;
        // These might work, but there's no guarantee it will work well.
        case COSINE_DISTANCE:
        case BHATTACHARYYA_METRIC: case BHATTACHARYYA_DISTANCE: case HELLINGER:

            return FULL_WEIGHTED_MEAN;

        case L2:  return GEO_MEDIAN;
        case L1:  return  L1_MEDIAN;
        case TVD: return TVD_MEDIAN;

    }
}

using util::l1_median;
using coresets::l1_median;

struct CentroidPolicy {
    template<typename VT, bool TF, typename Range, typename VT2=VT, typename RowSums>
    static void perform_average(blaze::DenseVector<VT, TF> &ret, const Range &r, const RowSums &rs,
                                const VT2 *wc = static_cast<VT2 *>(nullptr),
                                dist::DissimilarityMeasure measure=static_cast<dist::DissimilarityMeasure>(-1))
    {
        using FT = blz::ElementType_t<VT>;
        PREC_REQ(measure != static_cast<dist::DissimilarityMeasure>(-1), "Must define dissimilarity measure");
        if(measure == dist::TOTAL_VARIATION_DISTANCE) {
            if(wc)
                coresets::l1_median(r, ret, wc->data());
            else
                coresets::l1_median(r, ret);
        }
        else if(measure == dist::L1) {
            std::conditional_t<blz::IsSparseMatrix_v<Range>,
                               blz::CompressedMatrix<FT, blz::StorageOrder_v<Range> >,
                               blz::DynamicMatrix<FT, blz::StorageOrder_v<Range> >
            > cm = r % blz::expand(trans(rs), r.columns());
            if(wc)
                coresets::l1_median(cm, ret, wc->data());
            else
                coresets::l1_median(cm, ret);
        } else if(measure == dist::LLR || measure == dist::UWLLR || measure == dist::OLLR) {
            PRETTY_SAY << "LLR test\n";
            FT total_sum_inv;
            if(wc) {
                total_sum_inv = 1. / blz::dot(rs, *wc);
                *ret = blaze::sum<blz::columnwise>(r % blz::expand(trans(*wc * rs), r.columns())) * total_sum_inv;
            } else {
                total_sum_inv = 1. / blaze::sum(rs);
                *ret = blaze::sum<blz::columnwise>(r % blz::expand(trans(rs), r.columns())) * total_sum_inv;
            }
        } else if(wc) {
            PRETTY_SAY << "Weighted, anything but L1 or LLR (" << dist::detail::prob2str(measure) << ")\n";
            assert((*(*wc)).size() == r.rows());
            assert(blz::expand(*(*wc), r.columns()).rows() == r.rows());
            assert(blz::expand(*(*wc), r.columns()).columns() == r.columns());
            auto wsuminv = 1. / blaze::sum(*wc);
            if(!dist::detail::is_probability(measure)) { // e.g., take mean of unscaled values
                auto mat2schur = blz::expand(*(*wc) * rs, r.columns());
                PRETTY_SAY << "NOTPROB r dims: " << r.rows() << "/" << r.columns() << '\n';
                PRETTY_SAY << "NOTPROB mat2schur dims: " << mat2schur.rows() << "/" << mat2schur.columns() << '\n';
                *ret = blaze::sum<blz::columnwise>(r % blz::expand(*(*wc) * rs, r.columns())) * wsuminv;
            } else {                                    // Else take mean of scaled values
                auto mat2schur = blz::expand(*(*wc), r.columns());
                PRETTY_SAY << "PROB r dims: " << r.rows() << "/" << r.columns() << '\n';
                PRETTY_SAY << "PROB mat2schur dims: " << mat2schur.rows() << "/" << mat2schur.columns() << '\n';
                *ret = blaze::sum<blz::columnwise>(r % blz::expand(*(*wc), r.columns())) * wsuminv;
                assert(blaze::max(*ret) < 1. || !std::fprintf(stderr, "max in ret: %g for a probability distribution.", blaze::max(*ret)));
            }
        } else {
            PRETTY_SAY << "Unweighted, anything but L1 or LLR (" << dist::detail::prob2str(measure) << ")\n";
            if(dist::detail::is_probability(measure)) {
                // Weighted average for all
#ifndef NDEBUG
                auto expansion = blz::expand(trans(rs), r.columns());
                PRETTY_SAY << "PROB r dims: " << r.rows() << "/" << r.columns() << '\n';
                PRETTY_SAY << "NOTPROB expansion dims: " << expansion.rows() << "/" << expansion.columns() << '\n';
#endif
                *ret = blaze::sum<blz::columnwise>(r % blz::expand(trans(rs), r.columns())) * (1. / (blaze::sum(rs) * r.rows()));
            } else *ret = blz::mean<blz::columnwise>(r % blz::expand(trans(rs), r.columns()));
        }
    }
    template<typename Matrix, typename RSVec, typename PriorData=RSVec, typename FT=blz::ElementType_t<Matrix>, typename AsnV, typename WPT=blz::DV<FT, blz::rowVector>, bool WSO=blz::rowVector>
    static void perform_average(Matrix &mat, const RSVec &rs, std::vector<blz::DV<FT, blz::rowVector>> &centers,
                                AsnV &assignments, dist::DissimilarityMeasure measure,
                                const blaze::Vector<WPT, WSO> *weight_cv=nullptr, const PriorData *pd=nullptr)
    {
        // Scale weights up if necessary
        std::vector<blaze::SmallArray<uint32_t, 16>> assignv(centers.size());
        for(size_t i = 0; i < assignments.size(); ++i) {
            assert(assignments[i] < centers.size());
            assignv.at(assignments[i]).pushBack(i);
        }
        if(measure == dist::TVD || measure == dist::L1) {
            using ptr_t = decltype((**weight_cv).data());
            ptr_t ptr = nullptr;
            if(weight_cv) ptr = (**weight_cv).data();
            OMP_PFOR
            for(unsigned i = 0; i < centers.size(); ++i) {
                coresets::l1_median(mat, centers[i], assignv[i], ptr);
            }
            return;
        }
        [[maybe_unused]] auto pv = pd ? FT(pd->operator[](0)): FT(1.);
        assert(!pd || pd->size() == 1); // TODO: support varied prior over features
        for(unsigned i = 0; i < centers.size(); ++i) {
            auto aip = assignv[i].data();
            auto ain = assignv[i].size();
            auto r(blz::rows(mat, aip, ain));
            auto &c(centers[i]);

            if(weight_cv) {
                c = blaze::sum<blaze::columnwise>(
                    blz::rows(mat, aip, ain)
                    % blaze::expand(blaze::elements(trans(**weight_cv), aip, ain), mat.columns()));
            } else {
                c = blaze::sum<blaze::columnwise>(blz::rows(mat, aip, ain));
            }

            assert(rs.size() == mat.rows());
            if constexpr(blaze::IsSparseMatrix_v<Matrix>) {
                if(pd) {
                    if(weight_cv) {
                        c += pv * sum(blz::elements(rs * **weight_cv, aip, ain));
                    } else {
                        c += pv * ain;
                    }
                    for(const auto ri: assignv[i]) {
                        assert(ri < rs.size());
                        assert(ri < mat.rows());
                        auto rsri = pv;
                        if(!use_scaled_centers(measure)) rsri /= rs[ri];
                        for(const auto &pair: row(mat, ri, unchecked))
                            c[pair.index()] -= rsri;
                    }
                }
            }
            double div;
            if(measure == dist::LLR || measure == dist::OLLR || measure == dist::UWLLR) {
                if(weight_cv)
                    div = sum(blz::elements(rs * **weight_cv, aip, ain));
                else
                    div = sum(blz::elements(rs, aip, ain));
            } else {
                if(weight_cv) {
                    div = sum(**weight_cv);
                } else {
                    div = ain;
                }
            }
            auto oldnorm = blaze::l2Norm(c);
            c *= (1. / div);
            auto newnorm = blaze::l2Norm(c);
            assert(min(c) >= 0 || !std::fprintf(stderr, "min center loc: %g\n", min(c)));
        }
    }
    template<typename FT, typename Row, typename Src>
    static void __perform_increment(FT neww, FT cw, Row &ret, const Src &dat, FT row_sum, dist::DissimilarityMeasure measure)
    {
        if(measure == dist::L1 || measure == dist::TOTAL_VARIATION_DISTANCE)
            throw std::invalid_argument("__perform_increment is only for linearly-calculated means, not l1 median");
        if(cw == 0.) {
            if(dist::detail::is_probability(measure))
                ret = dat;
            else
                ret = dat * row_sum;
        } else {
            auto div = neww / (neww + cw);
            if(dist::detail::is_probability(measure)) {
                ret += (dat - ret) * div;
            } else if(measure == dist::LLR || measure == dist::UWLLR) {
                ret += (dat * row_sum) * neww;
                // Add up total sum and subtract later
                // since there are three weighting factors here:
                // First, partial assignment
                // Then point-wise weights (both of which are in neww)
                // Then, for LLR/UWLLR, there's weighting by the row-sums
            } else {
                // Maintain running mean for full vector value
                ret += (dat * row_sum - ret) * div;
            }
        }
    }

    template<typename VT, bool TF, typename RowSums, typename MatType, typename CenterCon, typename VT2=blz::DynamicVector<blz::ElementType_t<VT>> >
    static void perform_soft_assignment(const blz::DenseMatrix<VT, TF> &assignments,
        const RowSums &rs,
        OMP_ONLY(std::mutex *mutptr,)
        const MatType &data, CenterCon &newcon,
        const VT2 *wc = static_cast<const VT2 *>(nullptr),
        dist::DissimilarityMeasure measure=static_cast<dist::DissimilarityMeasure>(-1))
    {
        using FT = blz::ElementType_t<VT>;
        PREC_REQ(measure != static_cast<dist::DissimilarityMeasure>(-1), "Must define dissimilarity measure");
        if(measure == dist::L1 || measure == dist::TOTAL_VARIATION_DISTANCE) {
            OMP_PFOR
            for(unsigned j = 0; j < newcon.size(); ++j) {
                blz::DynamicVector<FT, blz::rowVector> newweights;
                {
                    auto col = trans(column(assignments, j));
                    if(wc) newweights = col * *wc;
                    else   newweights = col;
                }
                if(measure == dist::L1) {
                    std::conditional_t<blz::IsDenseMatrix_v<VT>,
                                       blz::DynamicMatrix<FT>, blz::CompressedMatrix<FT>>
                        scaled_data = data % blz::expand(rs, data.columns());
                    coresets::l1_median(scaled_data, newcon[j], newweights.data());
                } else { // TVD
                    coresets::l1_median(data, newcon[j], newweights.data());
                }
            }
        } else {
            blz::DynamicVector<FT> summed_contribs(newcon.size(), 0.);
            OMP_PFOR
            for(size_t i = 0; i < data.rows(); ++i) {
                auto item_weight = wc ? wc->operator[](i): static_cast<FT>(1.);
                const auto row_sum = rs[i];
                auto asn(row(assignments, i, unchecked));
                for(size_t j = 0; j < newcon.size(); ++j) {
                    auto &cw = summed_contribs[j];
                    if(auto asnw = asn[j]; asnw > 0.) {
                        auto neww = item_weight * asnw;
                        OMP_ONLY(if(mutptr) mutptr[j].lock();)
                        __perform_increment(neww, cw, newcon[j], row(data, i, unchecked), row_sum, measure);
                        OMP_ONLY(if(mutptr) mutptr[j].unlock();)
                        OMP_ATOMIC
                        cw += neww;
                    }
                }
            }
            if(measure == dist::LLR || measure == dist::UWLLR || measure == dist::OLLR) {
                OMP_PFOR
                for(auto i = 0u; i < newcon.size(); ++i)
                    newcon[i] *= 1. / blz::dot(column(assignments, i), rs);
            }
        }
    }
}; // CentroidPolicy


template<typename FT=double, typename Mat, typename AsnT, typename CostsT, typename CtrsT, typename WeightsT, typename IT=uint32_t>
void set_centroids_l1(const Mat &mat, AsnT &asn, CostsT &costs, CtrsT &ctrs, WeightsT *weights) {
    const unsigned k = ctrs.size();
    using asn_t = std::decay_t<decltype(asn[0])>;
    std::vector<std::vector<asn_t>> assigned(ctrs.size());
    assert(costs.size() == asn.size());
    for(size_t i = 0; i < costs.size(); ++i) {
        assert(asn[i] < assigned.size());
        blz::push_back(assigned[asn[i]], i);
    }
    blaze::SmallArray<asn_t, 16> sa;
    wy::WyRand<asn_t, 4> rng(costs.size());
    for(unsigned i = 0; i < k; ++i) if(assigned[i].empty()) blz::push_back(sa, i);
    while(!sa.empty()) {
        std::vector<uint32_t> idxleft;
        for(unsigned i = 0; i < k; ++i)
            if(std::find(sa.begin(), sa.end(), i) == sa.end())
                blz::push_back(idxleft, i);
        // Re-calculate for centers that have been removed
        for(auto idx: sa) {
            for(auto assigned_id: assigned[idx]) {
                auto ilit = idxleft.begin();
                auto myr = row(mat, assigned_id);
                auto fcost = l1Dist(ctrs[*ilit++], myr);
                asn_t bestid = 0;
                for(;ilit != idxleft.end();++ilit) {
                    auto ncost = l1Dist(ctrs[*ilit], myr);
                    if(ncost < fcost) bestid = *ilit, fcost = ncost;
                }
                costs[assigned_id] = fcost;
                asn[assigned_id] = bestid;
            }
        }
        // Use D2 sampling to re-seed
        for(const auto idx: sa) {
            std::ptrdiff_t found = reservoir_simd::sample(costs.data(), costs.size(), rng());
            set_center(ctrs[idx], row(mat, found));
            for(size_t i = 0; i < mat.rows(); ++i) {
                const auto c = l1Dist(ctrs[idx], row(mat, i, unchecked));
                if(c < costs[i]) {
                    asn[i] = idx;
                    costs[i] = c;
                }
            }
        }
        for(auto &subasn: assigned) subasn.clear();
        // Check for orphans again
        sa.clear();
        for(size_t i = 0; i < costs.size(); ++i) {
            blz::push_back(assigned[asn[i]], i);
        }
        for(const auto &subasn: assigned) if(subasn.empty()) sa.pushBack(&subasn - assigned.data());
    }
    for(unsigned i = 0; i < k; ++i) {
        const auto &asnv = assigned[i];
        const auto asp = asnv.data();
        const auto nasn = asnv.size();
        MINOCORE_VALIDATE(nasn != 0);
        switch(nasn) {
            case 1: set_center(ctrs[i], row(mat, asnv[0])); break;
            default: {
                if constexpr(blaze::IsMatrix_v<Mat>) {
                    auto rowsel = rows(mat, asp, nasn);
                    if(weights)
                        l1_median(rowsel, ctrs[i], elements(*weights, asp, nasn));
                    else
                        l1_median(rowsel, ctrs[i]);
                } else l1_median(mat, ctrs[i], asp, nasn, weights);
            } break;
        }
    }
}

using util::tvd_median;
using coresets::tvd_median;

template<typename FT=double, typename Mat, typename AsnT, typename CostsT, typename CtrsT, typename WeightsT, typename IT=uint32_t, typename RowSums>
void set_centroids_tvd(const Mat &mat, AsnT &asn, CostsT &costs, CtrsT &ctrs, WeightsT *weights, const RowSums &rsums) {
    const unsigned k = ctrs.size();
    using asn_t = std::decay_t<decltype(asn[0])>;
    std::vector<std::vector<asn_t>> assigned(ctrs.size());
    assert(costs.size() == asn.size());
    for(size_t i = 0; i < costs.size(); ++i) {
        assert(asn[i] < assigned.size());
        blz::push_back(assigned[asn[i]], i);
    }
    blaze::SmallArray<asn_t, 16> sa;
    wy::WyRand<asn_t, 4> rng(costs.size());
    for(unsigned i = 0; i < k; ++i) if(assigned[i].empty()) blz::push_back(sa, i);
    while(!sa.empty()) {
        std::vector<uint32_t> idxleft;
        for(unsigned i = 0; i < k; ++i)
            if(std::find(sa.begin(), sa.end(), i) == sa.end())
                blz::push_back(idxleft, i);
        // Re-calculate for centers that have been removed
        for(auto idx: sa) {
            for(auto assigned_id: assigned[idx]) {
                auto ilit = idxleft.begin();
                auto myr = row(mat, assigned_id);
                auto fcost = l1Dist(ctrs[*ilit++], myr);
                asn_t bestid = 0;
                for(;ilit != idxleft.end();++ilit) {
                    auto ncost = l1Dist(ctrs[*ilit], myr);
                    if(ncost < fcost) bestid = *ilit, fcost = ncost;
                }
                costs[assigned_id] = fcost;
                asn[assigned_id] = bestid;
            }
        }
        // Use D2 sampling to re-seed
        for(const auto idx: sa) {
            std::ptrdiff_t found = reservoir_simd::sample(costs.data(), costs.size(), rng());
            assert(found < (std::ptrdiff_t)(costs.size()));
            set_center(ctrs[idx], row(mat, found));
            for(size_t i = 0; i < mat.rows(); ++i) {
                const auto c = l1Dist(ctrs[idx], row(mat, i, unchecked));
                if(c < costs[i]) {
                    asn[i] = idx;
                    costs[i] = c;
                }
            }
        }
        for(auto &subasn: assigned) subasn.clear();
        // Check for orphans again
        sa.clear();
        for(size_t i = 0; i < costs.size(); ++i) {
            blz::push_back(assigned[asn[i]], i);
        }
        for(const auto &subasn: assigned) if(subasn.empty()) sa.pushBack(&subasn - assigned.data());
    }
    OMP_PFOR
    for(unsigned i = 0; i < k; ++i) {
        const auto &asnv = assigned[i];
        const auto asp = asnv.data();
        const auto nasn = asnv.size();
        MINOCORE_VALIDATE(nasn != 0);
        switch(nasn) {
            case 1: set_center(ctrs[i], row(mat, asnv[0])); break;
            default:
                tvd_median(mat, ctrs[i], asp, nasn, weights, rsums);
            break;
        }
    }
}

template<typename...Args>
void set_centroids_tvd(Args &&...args) {
    throw std::invalid_argument("TVD clustering not supported explicitly; instead, normalize your count vectors and perform the clustering with L1");
}

template<typename FT=double, typename Mat, typename AsnT, typename CostsT, typename CtrsT, typename WeightsT, typename IT=uint32_t>
void set_centroids_l2(const Mat &mat, AsnT &asn, CostsT &costs, CtrsT &ctrs, WeightsT *weights, double eps=0.) {
    using asn_t = std::decay_t<decltype(asn[0])>;
    std::vector<std::vector<asn_t>> assigned(ctrs.size());
    const size_t np = costs.size();
    const unsigned k = ctrs.size();
    for(size_t i = 0; i < np; ++i) {
        blz::push_back(assigned[asn[i]], i);
    }
    wy::WyRand<asn_t, 4> rng(costs.size());
    blaze::SmallArray<asn_t, 16> sa;
    for(unsigned i = 0; i < k; ++i) if(assigned[i].empty()) blz::push_back(sa, i);
    while(!sa.empty()) {
        // Compute partial sum
        std::vector<uint32_t> idxleft;
        for(unsigned int i = 0; i < k; ++i)
            if(std::find(sa.begin(), sa.end(), i) == sa.end())
                blz::push_back(idxleft, i);
        // Re-calculate for centers that have been removed
        for(auto idx: sa) {
            for(auto assigned_id: assigned[idx]) {
                auto ilit = idxleft.begin();
                auto myr = row(mat, assigned_id);
                auto fcost = l2Dist(ctrs[*ilit++], myr);
                asn_t bestid = 0;
                for(;ilit != idxleft.end();++ilit) {
                    auto ncost = l2Dist(ctrs[*ilit], myr);
                    if(ncost < fcost) bestid = *ilit, fcost = ncost;
                }
                costs[assigned_id] = fcost;
                asn[assigned_id] = bestid;
            }
        }
        // Use D2 sampling to re-seed
        for(const auto idx: sa) {
            std::ptrdiff_t found = reservoir_simd::sample(costs.data(), costs.size(), rng());
            set_center(ctrs[idx], row(mat, found));
            OMP_PFOR
            for(size_t i = 0; i < mat.rows(); ++i) {
                const auto c = l2Dist(ctrs[idx], row(mat, i, unchecked));
                if(c < costs[i]) {
                    asn[i] = idx;
                    costs[i] = c;
                }
            }
        }
        for(auto &subasn: assigned) subasn.clear();
        // Check for orphans again
        sa.clear();
        for(size_t i = 0; i < np; ++i) {
            blz::push_back(assigned[asn[i]], i);
        }
        for(const auto &subasn: assigned) if(subasn.empty()) sa.pushBack(&subasn - assigned.data());
    }
    for(unsigned i = 0; i < k; ++i) {
        const auto nasn = assigned[i].size();
        const auto asp = assigned[i].data();
        MINOCORE_VALIDATE(nasn != 0);
        if(nasn == 1) {
            set_center(ctrs[i], row(mat, *asp));
        } else {
            set_center_l2(ctrs[i], mat, asp, nasn, weights, eps);
#if 0
            auto rowsel = rows(mat, asp, nasn);
            VERBOSE_ONLY(std::cerr << "Calculating geometric median for " << nasn << " rows and storing in " << ctrs[i] << '\n';)
            if(weights)
                blz::geomedian(rowsel, ctrs[i], elements(*weights, asp, nasn), eps);
            else
                blz::geomedian(rowsel, ctrs[i], eps);
            VERBOSE_ONLY(std::cerr << "Calculated geometric median; new values: " << ctrs[i] << '\n';)
#endif
        }
    }
}

template<typename FT=double, typename Mat, typename PriorT, typename AsnT, typename CostsT, typename CtrsT, typename WeightsT, typename IT=uint32_t, typename SumT>
void set_centroids_full_mean(const Mat &mat,
    const dist::DissimilarityMeasure measure,
    const PriorT &prior, AsnT &asn, CostsT &costs, CtrsT &ctrs,
    WeightsT *weights, SumT &ctrsums, const SumT &rowsums)
{
    static_assert(std::is_floating_point_v<std::decay_t<decltype(ctrsums[0])>>, "SumT must be floating-point");
    assert(rowsums.size() == (*mat).rows());
    assert(ctrsums.size() == ctrs.size());
    DBG_ONLY(std::fprintf(stderr, "[%s] Calling set_centroids_full_mean with weights = %p\n", __PRETTY_FUNCTION__, (void *)weights);)
    //

    assert(asn.size() == costs.size() || !std::fprintf(stderr, "asn size %zu, cost size %zu\n", asn.size(), costs.size()));
    //using asn_t = std::decay_t<decltype(asn[0])>;
    blaze::SmallArray<size_t, 16> sa;
    wy::WyRand<size_t, 4> rng(costs.size()); // Used for restarting orphaned centers
    blz::DV<FT> pdf;
    const size_t np = costs.size(), k = ctrs.size();
    auto assigned = std::make_unique<std::vector<size_t>[]>(k);
    //set_asn:
    for(size_t i = 0; i < np; ++i) {
        assigned[asn[i]].push_back(i);
    }
#ifndef NDEBUG
    for(unsigned i = 0; i < k; ++i) std::fprintf(stderr, "Center %u has %zu assigned points\n", i, assigned[i].size());
#endif
    for(unsigned i = 0; i < k; ++i)
        if(assigned[i].empty())
            blz::push_back(sa, i);
#ifndef NDEBUG
    int nfails = 0;
    for(size_t i = 0; i < k; ++i) {
        //const auto manual_sum = std::accumulate(ctrs[i].begin(), ctrs[i].end(), 0., [](double x, auto &pair) {return x + pair.value();});
        //std::fprintf(stderr, "csum[%zu] (cached) %g, but calcualted: %g (via blaze) vs manual %g\n", i, ctrsums[i], sum(ctrs[i]), manual_sum);
        if(std::abs(ctrsums[i] - sum(ctrs[i])) > 1e-5) {
            ++nfails;
        }
    }
    assert(!nfails);
#endif
    if(const size_t ne = sa.size()) {
        char buf[256];
        const auto pv = prior.size() ? FT(prior[0]): FT(0);
        std::sprintf(buf, "Restarting centers with no support for set_centroids_full_mean: %s as measure with prior of size %zu (%g)\n",
                     msr2str(measure), prior.size(), pv);
        std::cerr << buf;
        const constexpr RestartMethodPol restartpol = RESTART_D2;
        const FT psum = prior.size() == 1 ? FT(prior[0]) * prior.size(): sum(prior);
        for(size_t i = 0; i < k; ++i)
            assigned[i].clear();
        std::vector<std::ptrdiff_t> rs;
        for(size_t i = 0; i < ne; ++i) {
            // Instead, use a temporary buffer to store partial sums and randomly select newly-started centers
            // for D2, and just ran
            std::ptrdiff_t r;
            if(restartpol == RESTART_GREEDY)
                r = reservoir_simd::argmax(costs, /*mt=*/true);
            else if(restartpol == RESTART_RANDOM)
                r = rng() % costs.size();
            else {
                assert(restartpol == RESTART_D2);
                r = reservoir_simd::sample(costs.data(), costs.size(), rng());
            }
            rs.push_back(r);
            const auto id = sa[i];
            set_center(ctrs[id], row(mat, r));
            ctrsums[id] = sum(ctrs[id]);
        }
        costs = std::numeric_limits<FT>::max();

        OMP_PFOR
        for(size_t i = 0; i < np; ++i) {
            unsigned bestid = 0;
            auto r = row(mat, i, unchecked);
            const auto rsum = rowsums[i];
            costs[i] = cmp::msr_with_prior(measure, r, ctrs[0], prior, psum, rsum, ctrsums[0]);
            assert(std::abs(sum(r) - rsum) < 1e-6 || !std::fprintf(stderr, "rsum %g and summed %g\n", rsum, sum(r)));
            for(unsigned j = 0; j < k; ++j) {
                const auto csum = ctrsums[j];
                DBG_ONLY(auto bsum = sum(ctrs[j]);)
                assert(std::abs(csum - bsum) < 1e-10 || !std::fprintf(stderr, "for k = %u, csum %g but found bsum %g\n", j, csum, bsum));
                const auto c = cmp::msr_with_prior(measure, r, ctrs[j], prior, psum,
                                                   rsum, csum);
                if(c < costs[i]) {
                    costs[i] = c, bestid = j;
                }
            }
            asn[i] = bestid;
            OMP_CRITICAL
            {
                assigned[bestid].push_back(i);
            }
        }
        for(size_t i = 0; i < ne; ++i) {
            auto pid = rs[i];
            const auto cid = sa[i];
            if(asn[pid] != cid) {
                asn[pid] = cid;
                costs[pid] = 0.;
                assigned[cid].push_back(pid);
            }
#if 0
            if(asn[rs[i]] != sa[i]) {
                DBG_ONLY(std::fprintf(stderr, "Point %zd is not assigned to itself (%zd, empty #%zd). Cost: %g. Cost with itself: %g\n", rs[i], sa[i], i, costs[rs[i]], cmp::msr_with_prior(measure, ctrs[sa[i]], ctrs[sa[i]], prior, psum, ctrsums[sa[i]], ctrsums[sa[i]]));)
                asn[rs[i]] = sa[i];
            }
#endif
        }
    }
#if defined(_OPENMP) && !BLAZE_USE_SHARED_MEMORY_PARALLELIZATION
    #pragma message("Parallelizing loop, may cause things to break")
    #pragma omp parallel for
#endif
    for(unsigned i = 0; i < k; ++i) {
        DBG_ONLY(std::fprintf(stderr, "Computing mean for centroid %u with %zu assigned points\n", i, assigned[i].size());)
        // Compute mean for centroid
        const auto nasn = assigned[i].size();
        const auto asp = assigned[i].data();
        auto &ctr = ctrs[i];
        if(nasn == 1) {
            auto mr = row(mat, *asp);
            assert(ctr.size() == mr.size());
            set_center(ctr, mr);
        } else if(nasn == 0) {
            // do nothing
        } else {
            set_center(ctr, mat, asp, nasn, weights);
        }
    }
}

template<typename Vector, typename AT, bool ATF>
INLINE void correct_softmax(const Vector &costs, blaze::Vector<AT, ATF> &asn) {
    using CT = std::common_type_t<blz::ElementType_t<Vector>, blz::ElementType_t<AT>>;
    using FT = std::conditional_t<std::is_floating_point_v<CT>, CT, std::conditional_t<(sizeof(CT) <= 4), float, double>>;
    if(isnan(*asn)) {
        auto bestind = reservoir_simd::argmin(costs.data(), costs.size(), /*mt=*/false);
        blaze::SmallArray<uint32_t, 8> sa;
        for(unsigned i = 0; i < costs.size(); ++i) if(costs[i] == costs[bestind]) sa.pushBack(i);
        FT per = 1. / sa.size();
        (*asn).reset();
        for(const auto ind: sa) (*asn)[ind] = per;
    }
}

template<typename FT=double, typename Mat, typename PriorT, typename CostsT, typename CtrsT, typename WeightsT, typename IT=uint32_t, typename SumT>
double set_centroids_full_mean(const Mat &mat,
    const dist::DissimilarityMeasure measure,
    const PriorT &, CostsT &costs, CostsT &asns, CtrsT &ctrs,
    WeightsT *weights, FT temp, SumT &ctrsums)
{
    assert(ctrsums.size() == ctrs.size());
    //std::fprintf(stderr, "Calling set_centroids_full_mean with weights = %p, temp = %g\n", (void *)weights, temp);

    const unsigned k = ctrs.size();
    //blz::DV<FT, blz::rowVector> wsums(k, 0.), asn(k, 0.);
    asns = softmax<rowwise>(costs * -temp);
    //std::cerr << softmaxcosts << '\n';
    //std::fprintf(stderr, "costs are of dim %zu/%zu\n", softmaxcosts.rows(), softmaxcosts.columns());
    double ret = 0.;
    OMP_PRAGMA("omp parallel for reduction(+:ret)")
    for(size_t i = 0; i < asns.rows(); ++i) {
        auto cr = row(costs, i, unchecked);
        auto r = row(asns, i, unchecked);
        //std::cerr << "costs: " << cr << '\n' << r << '\n';
        correct_softmax(cr, r);
        ret += dot(cr, r) * (weights ? double((*weights)[i]): 1.);
        //std::cerr << "post corrected: " << cr << '\n' << r << '\n';
    }
    //std::fprintf(stderr, "Sum of costs: %g\n", ret);
    //std::fprintf(stderr, "Now setting centers\n");
    if(measure == distance::L2 || measure == distance::L1) {
        //OMP_PFOR
        for(size_t i = 0; i < k; ++i) {
            blz::DV<FT, blz::columnVector> colweights;
            if(!weights) {
                colweights = column(asns, i, unchecked);
            } else if constexpr(blz::TransposeFlag_v<WeightsT> == blz::columnVector) {
                colweights = *weights * column(asns, i, unchecked);
            } else {
                colweights = trans(*weights) * column(asns, i, unchecked);
            }
            if(measure == distance::L2) {
                //std::fprintf(stderr, "l2 geomedian\n");
                if constexpr(blaze::IsMatrix_v<Mat>) {
                    geomedian(mat, ctrs[i], colweights.data());
                } else {
                    geomedian(mat, ctrs[i], (uint64_t *)nullptr, 0, &colweights);
                }
            } else {
                //std::fprintf(stderr, "l1median\n");
                l1_median(mat, ctrs[i], (uint64_t *)nullptr, 0, &colweights);
            }
        }
    } else { // full weighted mean (Bregman)
        if(weights) {
            std::fprintf(stderr, "weights: %p\n", (void *)weights);
            blz::DV<FT, rowVector> wsums;
            if constexpr(blz::TransposeFlag_v<WeightsT> == blz::columnVector) {
                wsums = 1. / sum<columnwise>(asns % expand(trans(*weights), asns.columns()));
            } else {
                wsums = 1. / sum<columnwise>(asns % expand(*weights, asns.columns()));
            }
            std::cerr << wsums << '\n';
            for(size_t i = 0; i < k; ++i) {
                if constexpr(blaze::TransposeFlag_v<WeightsT> != blaze::TransposeFlag_v<decltype(column(asns, i))>) {
                    ctrs[i] = (blz::sum<blz::columnwise>(mat % expand(column(asns, i) * trans(*weights), mat.columns())) * wsums[i]);
                } else {
                    ctrs[i] = blz::sum<blz::columnwise>(mat % expand(column(asns, i) * *weights, mat.columns())) * wsums[i];
                }
            }
        } else {
            blz::DV<FT> wsums = trans(1. / sum<columnwise>(asns));
            std::cerr << wsums << '\n';
            for(size_t i = 0; i < k; ++i) {
                auto expmat = expand(column(asns, i), mat.columns());
                ctrs[i] = (blz::sum<blz::columnwise>(mat % expmat) * wsums[i]);
                assert(ctrs[i].size() == mat.columns());
            }
        }
    }
    OMP_PFOR
    for(size_t i = 0; i < k; ++i) ctrsums[i] = sum(ctrs[i]);
    DBG_ONLY(std::fprintf(stderr, "Centroids set, soft, with T = %g. Center sums: \n\n", temp);)
    return ret;
}

template<typename FT=double, typename VT, typename IT, typename IPtrT, typename PriorT, typename CostsT, typename CtrsT, typename WeightsT, typename SumT>
double set_centroids_full_mean(const util::CSparseMatrix<VT, IT, IPtrT> &mat,
    const dist::DissimilarityMeasure measure,
    const PriorT &, CostsT &costs, CostsT &asns, CtrsT &ctrs,
    WeightsT *weights, FT temp, SumT &ctrsums)
{
    assert(ctrsums.size() == ctrs.size());
    DBG_ONLY(std::fprintf(stderr, "Calling set_centroids_full_mean with weights = %p, temp = %g\n", (void *)weights, temp);)
    assert(min(costs) >= 0. || !std::fprintf(stderr, "mincost: %g\n", min(costs)));
    asns = softmax<rowwise>(costs * -temp);
    double ret = 0.;
    OMP_PRAGMA("omp parallel for reduction(+:ret)")
    for(size_t i = 0; i < asns.rows(); ++i) {
        auto r(row(asns, i, unchecked));
        auto cr(row(costs, i, unchecked));
        correct_softmax(cr, r);
        const double w = weights ? double((*weights)[i]): 1.;
        ret += dot(cr, r) * w;
    }
    std::vector<blz::DV<FT>> tmprows(ctrs.size(), blz::DV<FT>(mat.columns(), 0.));
    if(measure == distance::L2 || measure == distance::L1) {
        //OMP_PFOR
        for(size_t i = 0; i < ctrs.size(); ++i) {
            blz::DV<FT, blz::columnVector> colweights;
            if(!weights) {
                colweights = column(asns, i, unchecked);
            } else if constexpr(blz::TransposeFlag_v<WeightsT> == blz::columnVector) {
                colweights = *weights * column(asns, i, unchecked);
            } else {
                colweights = trans(*weights) * column(asns, i, unchecked);
            }
            std::fprintf(stderr, "Weights selected for row %zu/%zu\n", i + 1, ctrs.size());
            uint64_t *np = 0;
            if(measure == distance::L2)
                geomedian(mat, tmprows[i], np, 0, &colweights);
            else
                l1_median(mat, tmprows[i], np, 0, &colweights);
            std::fprintf(stderr, "Centroid selected for row %zu/%zu\n", i + 1, ctrs.size());
            if constexpr(blz::TransposeFlag_v<std::decay_t<decltype(ctrs[0])>> == blaze::rowVector) {
                ctrs[i] = trans(tmprows[i]);
            } else {
                ctrs[i] = tmprows[i];
            }
        }
    } else {
        std::fprintf(stderr, "Start vector\n");
        blz::DV<FT, columnVector> winv;
        if(weights) {
            if constexpr(blz::TransposeFlag_v<WeightsT> == rowVector) {
                winv = trans(1. / (*weights * asns));
            } else {
                winv = 1. / trans((trans(*weights) * asns));
            }
        } else {
            winv = 1. / trans(sum<columnwise>(asns));
        }
        OMP_PFOR
        for(size_t j = 0; j < mat.rows(); ++j) {
            auto r = row(mat, j, unchecked);
            auto smr = row(asns, j, unchecked);
            for(size_t i = 0; i < r.n_; ++i) {
                auto data = r.data_[i];
                auto idx = r.indices_[i];
                size_t m = 0;
                for(; m < ctrs.size(); ++m) {
                    OMP_ATOMIC
                    tmprows[m][idx] += smr[m] * data;
                }
            }
        }
        OMP_PFOR
        for(size_t i = 0; i < tmprows.size(); ++i) {
            if constexpr(blz::TransposeFlag_v<std::decay_t<decltype(ctrs[0])>> == blaze::rowVector) {
                ctrs[i] = trans(tmprows[i] * winv[i]);
            } else {
                ctrs[i] = tmprows[i] * winv[i];
            }
        }
    }
    OMP_PFOR
    for(size_t i = 0; i < ctrs.size(); ++i) ctrsums[i] = sum(ctrs[i]);
    //for(const auto &ctr: ctrs) std::cerr << ctr << '\n';
    DBG_ONLY(std::fprintf(stderr, "Centroids set, soft, with T = %g\n", temp);)
    return ret;
}

} } // namespace minicore::clustering

#endif /* MINOCORE_CLUSTERING_CENTROID_H__ */
