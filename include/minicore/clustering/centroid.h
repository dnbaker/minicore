#ifndef MINOCORE_CLUSTERING_CENTROID_H__
#define MINOCORE_CLUSTERING_CENTROID_H__
#include "minicore/util/blaze_adaptor.h"
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
    blz::DV<VT> mv(mat.columns(), VT(0));
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
    wsum = 1. / (w ? wsum: double(nasn));
    ctr.reset();
    for(size_t i = 0; i < mv.size(); ++i) {
        if(mv[i] > 0.) {
            ctr[i] = mv[i];
        }
    }
    ctr *= wsum;
}
template<typename CtrT, typename DataT, typename IndicesT, typename IndPtrT, typename IT, typename WeightT>
void set_center_l2(CtrT &center, const util::CSparseMatrix<DataT, IndicesT, IndPtrT> &mat, IT *asp, size_t nasn, WeightT *weights, double eps) {
    util::geomedian(mat, center, asp, nasn, weights, eps);
}

template<typename CtrT, typename MT, typename IT, typename WeightT>
void set_center_l2(CtrT &center, const blaze::Matrix<MT, blaze::rowMajor> &mat, IT *asp, size_t nasn, WeightT *weights, double eps) {
    auto rowsel = rows(mat, asp, nasn);
    VERBOSE_ONLY(std::cerr << "Calculating geometric median for " << nasn << " rows and storing in " << center << '\n';)
    if(weights)
        blz::geomedian(rowsel, center, elements(*weights, asp, nasn), eps);
    else
        blz::geomedian(rowsel, center, eps);
    VERBOSE_ONLY(std::cerr << "Calculated geometric median; new values: " << ctrs[i] << '\n';)
}

template<typename CtrT, typename MT, bool SO, typename IT, typename WeightT=blz::DV<blz::ElementType_t<MT>>>
void set_center(CtrT &ctr, const blaze::Matrix<MT, SO> &mat, IT *asp, size_t nasn, WeightT *w = static_cast<WeightT>(nullptr)) {
    auto rowsel = rows(mat, asp, nasn);
    if(w) {
        auto elsel = elements(*w, asp, nasn);
        auto weighted_rows = rowsel % blaze::expand(elsel, mat.columns());
        // weighted sum over total weight -> weighted mean
        ctr = blaze::sum<blaze::columnwise>(weighted_rows) / blaze::sum(elsel);
    } else ctr = blaze::mean<blaze::columnwise>(rowsel);
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

struct CentroidPolicy {
    template<typename VT, bool TF, typename Range, typename VT2=VT, typename RowSums>
    static void perform_average(blaze::DenseVector<VT, TF> &ret, const Range &r, const RowSums &rs,
                                const VT2 *wc = static_cast<VT2 *>(nullptr),
                                dist::DissimilarityMeasure measure=static_cast<dist::DissimilarityMeasure>(-1))
    {
        using FT = blz::ElementType_t<VT>;
        PREC_REQ(measure != static_cast<dist::DissimilarityMeasure>(-1), "Must define dissimilarity measure");
        if(measure == dist::TOTAL_VARIATION_DISTANCE) {
            PRETTY_SAY << "TVD: performing " << (wc ? static_cast<const char *>("weighted"): static_cast<const char *>("unweighted")) << "L1 median on *normalized* categorical distributions.\n";
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
            PRETTY_SAY << "L1: performing " << (wc ? static_cast<const char *>("weighted"): static_cast<const char *>("unweighted")) << "L1 median on *unnormalized* categorical distributions, IE absolute count data.\n";
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
            std::fprintf(stderr, "Setting center %u with %zu support\n", i, ain);
            auto r(blz::rows(mat, aip, ain));
            auto &c(centers[i]);
            std::fprintf(stderr, "max row index: %u\n", *std::max_element(aip, aip + ain));

            if(weight_cv) {
                c = blaze::sum<blaze::columnwise>(
                    blz::rows(mat, aip, ain)
                    % blaze::expand(blaze::elements(trans(**weight_cv), aip, ain), mat.columns()));
            } else {
                std::fprintf(stderr, "Performing unweighted sum of %zu rows\n", ain);
                c = blaze::sum<blaze::columnwise>(blz::rows(mat, aip, ain));
            }

            assert(rs.size() == mat.rows());
            if constexpr(blaze::IsSparseMatrix_v<Matrix>) {
                if(pd) {
                    std::fprintf(stderr, "Sparse prior handling\n");
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
                    std::fprintf(stderr, "weighted, nonLLR\n");
                    div = sum(**weight_cv);
                } else {
                    std::fprintf(stderr, "unweighted, nonLLR\n");
                    div = ain;
                }
            }
            auto oldnorm = blaze::l2Norm(c);
            c *= (1. / div);
            auto newnorm = blaze::l2Norm(c);
            std::fprintf(stderr, "norm before %g, after %g\n", oldnorm, newnorm);
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
    std::unique_ptr<blz::DV<FT>> probup;
    assert(costs.size() == asn.size());
    for(size_t i = 0; i < costs.size(); ++i) {
        assert(asn[i] < assigned.size());
        blz::push_back(assigned[asn[i]], i);
    }
    blaze::SmallArray<asn_t, 16> sa;
    wy::WyRand<asn_t, 4> rng(costs.size());
    for(unsigned i = 0; i < k; ++i) if(assigned[i].empty()) blz::push_back(sa, i);
    while(!sa.empty()) {
        // Compute partial sum
#ifndef NDEBUG
        std::fprintf(stderr, "reseeding %zu centers\n", sa.size());
#endif
        if(!probup) probup.reset(new blz::DV<FT>(mat.rows()));
        FT *pd = probup->data(), *pe = pd + probup->size();
        auto cb = costs.begin(), ce = costs.end();
        if(weights) {
            std::partial_sum(cb, ce, pd, [ds=&costs[0],&weights](auto x, const auto &y) {
                return x + y * ((*weights)[&y - ds]);
            });
        } else std::partial_sum(cb, ce, pd);
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
            std::uniform_real_distribution<double> dist;
            std::ptrdiff_t found = std::lower_bound(pd, pe, dist(rng) * pe[-1]) - pd;
            assert(found < (std::ptrdiff_t)(pe - pd));
            assign(ctrs[idx], row(mat, found));
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
            case 1: assign(ctrs[i], row(mat, asnv[0])); break;
            default: {
                if constexpr(blaze::IsMatrix_v<Mat>) {
                    auto rowsel = rows(mat, asp, nasn);
                    if(weights)
                        coresets::l1_median(rowsel, ctrs[i], elements(*weights, asp, nasn));
                    else
                        coresets::l1_median(rowsel, ctrs[i]);
                } else coresets::l1_median(mat, ctrs[i], asp, nasn, weights);
            } break;
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
    std::unique_ptr<blz::DV<FT>> probup;
    for(size_t i = 0; i < np; ++i) {
        blz::push_back(assigned[asn[i]], i);
    }
    wy::WyRand<asn_t, 4> rng(costs.size());
    blaze::SmallArray<asn_t, 16> sa;
    for(unsigned i = 0; i < k; ++i) if(assigned[i].empty()) blz::push_back(sa, i);
    while(!sa.empty()) {
        // Compute partial sum
#ifndef NDEBUG
        std::fprintf(stderr, "reseeding %zu centers\n", sa.size());
#endif
        if(!probup) probup.reset(new blz::DV<FT>(mat.rows()));
        FT *pd = probup->data(), *pe = pd + probup->size();
        if(weights) {
            ::std::partial_sum(costs.begin(), costs.end(), pd, [&weights,ds=&costs[0]](auto x, const auto &y) {
                return x + y * ((*weights)[&y - ds]);
            });
        } else {
            std::partial_sum(costs.begin(), costs.end(), pd);
        }
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
            std::uniform_real_distribution<double> dist;
            std::ptrdiff_t found = std::lower_bound(pd, pe, dist(rng) * pe[-1]) - pd;
            assert(found < (std::ptrdiff_t)(pe - pd));
            assign(ctrs[idx], row(mat, found));
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
            util::assign(ctrs[i], row(mat, *asp));
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
    assert(rowsums.size() == (*mat).rows());
    assert(ctrsums.size() == ctrs.size());
    DBG_ONLY(std::fprintf(stderr, "Calling set_centroids_full_mean with weights = %p\n", (void *)weights);)
    //

    assert(asn.size() == costs.size() || !std::fprintf(stderr, "asn size %zu, cost size %zu\n", asn.size(), costs.size()));
    using asn_t = std::decay_t<decltype(asn[0])>;
    blaze::SmallArray<asn_t, 16> sa;
    wy::WyRand<asn_t, 4> rng(costs.size()); // Used for restarting orphaned centers
    blz::DV<FT> pdf;
    const size_t np = costs.size(), k = ctrs.size();
    std::vector<std::vector<asn_t>> assigned(k);
    std::unique_ptr<blz::DV<FT>> probup;
    set_asn:
    for(size_t i = 0; i < np; ++i) {
        assigned[asn[i]].push_back(i);
    }
#ifndef NDEBUG
    for(unsigned i = 0; i < assigned.size(); ++i) std::fprintf(stderr, "Center %u has %zu assigned points\n", i, assigned[i].size());
#endif
    for(unsigned i = 0; i < k; ++i)
        if(assigned[i].empty())
            blz::push_back(sa, i);
    if(sa.size()) {
        std::fprintf(stderr, "make sa. sa size: %zu\n", sa.size());
        char buf[256];
        const auto pv = prior.size() ? FT(prior[0]): FT(0);
        std::sprintf(buf, "Restarting centers with no support for set_centroids_full_mean: %s as measure with prior of size %zu (%g)\n",
                     msr2str(measure), prior.size(), pv);
        std::cerr << buf;
        const constexpr RestartMethodPol restartpol = RESTART_D2;
        const FT psum = prior.size() == 1 ? FT(prior[0]) * prior.size(): sum(prior);
        for(const auto id: sa) {
            // Instead, use a temporary buffer to store partial sums and randomly select newly-started centers
            // for D2, and just ran
            std::ptrdiff_t r;
            if(restartpol == RESTART_GREEDY)
                r = reservoir_simd::argmax(costs, /*mt=*/true);
            else if(restartpol == RESTART_RANDOM)
                r = rng() % costs.size();
            else {
                assert(restartpol == RESTART_D2);
                int i = 0;
                do {
                    r = reservoir_simd::sample(costs.data(), costs.size(), rng());
                    std::fprintf(stderr, "Restarting with point %ld\n", r);
                    if(++i == 5) {
                        r = reservoir_simd::argmax(costs, true);
                        std::fprintf(stderr, "Center restarting took too many tries\n");
                    }
                } while(!costs[r]);
            }
            auto &ctr = ctrs[id];
            util::assign(ctr, row(mat, r));
            ctrsums[id] = sum(ctr);
            costs = std::numeric_limits<FT>::max();
            OMP_PFOR
            for(size_t i = 0; i < np; ++i) {
                unsigned bestid = asn[i], obi = bestid;
                auto r = row(mat, i, unchecked);
                const auto rsum = rowsums[i];
                assert(std::abs(sum(r) - rsum) < 1e-6 || !std::fprintf(stderr, "rsum %g and summed %g\n", rsum, sum(r)));
                for(unsigned j = 0; j < k; ++j) {
                    const auto csum = ctrsums[j];
                    DBG_ONLY(auto bsum = sum(ctrs[j]);)
                    assert(std::abs(csum - bsum) < 1e-10 || !std::fprintf(stderr, "csum %g but found bsum %g\n", csum, bsum));
                    const auto c = cmp::msr_with_prior(measure, r, ctrs[j], prior, psum,
                                                       rsum, csum);
                    if(c < costs[i]) {
                        costs[i] = c, bestid = j;
                    }
                }
                if(bestid != obi)
                    asn[i] = bestid;
            }
        }
        sa.clear();
        for(auto &x: assigned) x.clear();
        goto set_asn;
    }
#if defined(_OPENMP) && !BLAZE_USE_SHARED_MEMORY_PARALLELIZATION
    #pragma message("Parallelizing loop, may cause things to break")
    #pragma omp parallel for
#else
#pragma message("Not parallelizing loop")
#endif
    for(unsigned i = 0; i < k; ++i) {
        // Compute mean for centroid
        const auto nasn = assigned[i].size();
        const auto asp = assigned[i].data();
        auto &ctr = ctrs[i];
        if(nasn == 1) {
            util::assign(ctr, row(mat, *asp));
        } else {
            set_center(ctr, mat, asp, nasn, weights);
        }
        ctrsums[i] = sum(ctr);
    }
}

template<typename FT=double, typename Mat, typename PriorT, typename CostsT, typename CtrsT, typename WeightsT, typename IT=uint32_t, typename SumT>
void set_centroids_full_mean(const Mat &mat,
    const dist::DissimilarityMeasure,
    const PriorT &, CostsT &costs, CtrsT &ctrs,
    WeightsT *weights, FT temp, SumT &ctrsums)
{
    assert(ctrsums.size() == ctrs.size());
    std::fprintf(stderr, "Calling set_centroids_full_mean with weights = %p, temp = %g\n", (void *)weights, temp);
#ifndef NDEBUG
    for(size_t i = 0; i < ctrs.size(); ++i) {
        auto s = sum(ctrs[i]);
        assert(std::abs(s - ctrsums[i]) <= 1e-4 || !std::fprintf(stderr, "sum expected %g, got %g\n", s, ctrsums[i]));
    }
#endif

    const unsigned k = ctrs.size();
    blz::DV<FT, blz::rowVector> wsums(k, 0.), asn(k, 0.);
    OMP_PFOR
    for(unsigned i = 0; i < ctrs.size(); ++i) ctrs[i].reset(); // set to 0
    // Currently, this locks each center uniquely
    // This is not ideal, but it's hard to handle this atomically
    // TODO: provide a better parallelization method, either
    // 1. reduction
    // 2. more fine-grained locking strategies (though would fail for low-dimension data)
#if !BLAZE_USE_SHARED_MEMORY_PARALLELIZATION && defined(_OPENMP)
    auto locks = std::make_unique<std::mutex[]>(k);
    #pragma omp parallel for
#endif
    for(uint64_t i = 0; i < costs.rows(); ++i) {
        const auto r = row(costs, i, unchecked);
        const auto mr = row(mat, i, unchecked);
        const FT w = weights ? weights->operator[](i): FT(1);
        assert(asn.size() == r.size());
        asn = softmax(r * -temp);
#if !BLAZE_USE_SHARED_MEMORY_PARALLELIZATION && defined(_OPENMP)
#define WSUMINC(ind) do {OMP_ATOMIC wsums[ind] += w; } while(0)
#define GETLOCK(ind) std::lock_guard<std::mutex> lock(locks[ind])
#define __MT 0
#else
#define WSUMINC(ind) do {wsums[ind] += w;} while(0)
#define GETLOCK(ind)
#define __MT 1
#endif

#if VERBOSE_AF
        std::cerr << "assignments are " << asn << " from costs " << r << '\n';
#endif
        if(isnan(asn)) {
            auto bestind = reservoir_simd::argmin(r, __MT);
            asn.reset();
            asn[bestind] = 1.;
            WSUMINC(bestind);
            GETLOCK(bestind);
            ctrs[bestind] += mr * w;
        } else {
            for(unsigned j = 0; j < k; ++j) {
                const auto aiv = asn[j];
                if(aiv == 0.) continue;
                WSUMINC(j);
                GETLOCK(j);
                ctrs[j] += mr * (aiv * w);
            }
        }
    }
#undef GETLOCK
#undef WSUMINC
#undef __MT
    DBG_ONLY(std::fprintf(stderr, "About to set centers in parallel.\n");)
    OMP_PFOR
    for(unsigned j = 0; j < k; ++j) {
        ctrs[j] /= wsums[j];
        ctrsums[j] = sum(ctrs[j]);
    }
    DBG_ONLY(std::fprintf(stderr, "Centroids set, soft, with T = %g\n", temp);)
}

} } // namespace minicore::clustering

#endif /* MINOCORE_CLUSTERING_CENTROID_H__ */
