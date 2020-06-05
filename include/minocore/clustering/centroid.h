#ifndef MINOCORE_CLUSTERING_CENTROID_H__
#define MINOCORE_CLUSTERING_CENTROID_H__
#include "minocore/dist.h"
#include "minocore/util/blaze_adaptor.h"
#include "minocore/optim/kmedian.h"

namespace minocore { namespace clustering {

#if 0
template<typename Mat, typename RowSums>
struct RAIIScaler {
    Mat &m_;
    const RowSums &rs_;
    RAIIScaler(Mat &m, const RowSums &rs): m_(m), rs_(rs) {
        m_ %= blz::expand(trans(rs_), m_.columns());
    }
    ~RAIIScaler() {
        m_ %= blz::expand(trans(1. / rs_), m_.columns());
    }
};

template<typename Mat, typename RowSums>
auto make_raiiscaler(Mat &m, const RowSums &rs) {
    return RAIIScaler<Mat, RowSums>(m, rs);
}

#endif

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
                ~ret = blaze::sum<blz::columnwise>(r % blz::expand(trans(*wc * rs), r.columns())) * total_sum_inv;
            } else {
                total_sum_inv = 1. / blaze::sum(rs);
                ~ret = blaze::sum<blz::columnwise>(r % blz::expand(trans(rs), r.columns())) * total_sum_inv;
            }
        } else if(wc) {
            PRETTY_SAY << "Weighted, anything but L1 or LLR (" << dist::detail::prob2str(measure) << ")\n";
            assert((~(*wc)).size() == r.rows());
            assert(blz::expand(~(*wc), r.columns()).rows() == r.rows());
            assert(blz::expand(~(*wc), r.columns()).columns() == r.columns());
            auto wsuminv = 1. / blaze::sum(*wc);
            if(!dist::detail::is_probability(measure)) { // e.g., take mean of unscaled values
                auto mat2schur = blz::expand(~(*wc) * rs, r.columns());
                PRETTY_SAY << "NOTPROB r dims: " << r.rows() << "/" << r.columns() << '\n';
                PRETTY_SAY << "NOTPROB mat2schur dims: " << mat2schur.rows() << "/" << mat2schur.columns() << '\n';
                ~ret = blaze::sum<blz::columnwise>(r % blz::expand(~(*wc) * rs, r.columns())) * wsuminv;
            } else {                                    // Else take mean of scaled values
                auto mat2schur = blz::expand(~(*wc), r.columns());
                PRETTY_SAY << "PROB r dims: " << r.rows() << "/" << r.columns() << '\n';
                PRETTY_SAY << "PROB mat2schur dims: " << mat2schur.rows() << "/" << mat2schur.columns() << '\n';
                ~ret = blaze::sum<blz::columnwise>(r % blz::expand(~(*wc), r.columns())) * wsuminv;
                assert(blaze::max(~ret) < 1. || !std::fprintf(stderr, "max in ret: %g for a probability distribution.", blaze::max(~ret)));
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
                ~ret = blaze::sum<blz::columnwise>(r % blz::expand(trans(rs), r.columns())) * (1. / (blaze::sum(rs) * r.rows()));
            } else ~ret = blz::mean<blz::columnwise>(r % blz::expand(trans(rs), r.columns()));
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
            using ptr_t = decltype((~*weight_cv).data());
            ptr_t ptr = nullptr;
            if(weight_cv) ptr = (~*weight_cv).data();
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
                    % blaze::expand(blaze::elements(trans(~*weight_cv), aip, ain), mat.columns()));
            } else {
                std::fprintf(stderr, "Performing unweighted sum of %zu rows\n", ain);
                c = blaze::sum<blaze::columnwise>(blz::rows(mat, aip, ain));
            }

            assert(rs.size() == mat.rows());
            if constexpr(blaze::IsSparseMatrix_v<Matrix>) {
                if(pd) {
                    std::fprintf(stderr, "Sparse prior handling\n");
                    if(weight_cv) {
                        c += pv * blz::sum(blz::elements(rs * ~*weight_cv, aip, ain));
                    } else {
                        c += pv * ain;
                    }
                    for(const auto ri: assignv[i]) {
                        assert(ri < rs.size());
                        assert(ri < mat.rows());
                        auto rsri = pv;
                        if(!use_scaled_centers(measure)) rsri /= rs[ri];
                        for(const auto &pair: row(mat, ri, blz::unchecked))
                            c[pair.index()] -= rsri;
                    }
                }
            }
            double div;
            if(measure == dist::LLR || measure == dist::OLLR || measure == dist::UWLLR) {
                if(weight_cv)
                    div = blz::sum(blz::elements(rs * ~*weight_cv, aip, ain));
                else
                    div = blz::sum(blz::elements(rs, aip, ain));
            } else {
                if(weight_cv) {
                    std::fprintf(stderr, "weighted, nonLLR\n");
                    div = blz::sum(~*weight_cv);
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
                auto asn(row(assignments, i, blz::unchecked));
                for(size_t j = 0; j < newcon.size(); ++j) {
                    auto &cw = summed_contribs[j];
                    if(auto asnw = asn[j]; asnw > 0.) {
                        auto neww = item_weight * asnw;
                        OMP_ONLY(if(mutptr) mutptr[j].lock();)
                        __perform_increment(neww, cw, newcon[j], row(data, i, blz::unchecked), row_sum, measure);
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

} } // namespace minocore::clustering

#endif /* MINOCORE_CLUSTERING_CENTROID_H__ */
