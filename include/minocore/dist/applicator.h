#ifndef FGC_JSD_H__
#define FGC_JSD_H__
#include "minocore/util/exception.h"
#include "minocore/util/merge.h"
#include "minocore/coreset.h"
#include "minocore/dist/distance.h"
#include "distmat/distmat.h"
#include "minocore/optim/kmeans.h"
#include "minocore/util/csc.h"
#include <set>


namespace minocore {

namespace cmp {

using namespace blz;
using namespace blz::distance;


template<typename MatrixType, typename ElementType=blaze::ElementType_t<MatrixType>>
class DissimilarityApplicator {
    static constexpr bool IS_CSC_VIEW    = is_csc_view_v<MatrixType>;
    static constexpr bool IS_SPARSE      = IsSparseMatrix_v<MatrixType> || IS_CSC_VIEW;
    static constexpr bool IS_DENSE_BLAZE = IsDenseMatrix_v<MatrixType>;

    using ET = ElementType;

    MatrixType &data_;
    using VecT = blaze::DynamicVector<ET, IsRowMajorMatrix_v<MatrixType> || is_csc_view_v<MatrixType> ? blaze::rowVector: blaze::columnVector>;
    using matrix_type = MatrixType;
    VecT row_sums_;
    using CacheMatrixType = std::conditional_t<IS_SPARSE, blz::SM<ET>, blz::DM<ET> >;
    CacheMatrixType logdata_;
    CacheMatrixType sqrdata_;
    VecT jsd_cache_;
public:
    std::unique_ptr<VecT> prior_data_;
    double prior_sum_ = 0.;
    std::unique_ptr<VecT> l2norm_cache_;
    std::unique_ptr<VecT> pl2norm_cache_;
    using FT = ET;
    using MT = MatrixType;
    using This = DissimilarityApplicator<MatrixType>;
    using ConstThis = const DissimilarityApplicator<MatrixType>;
    static_assert(std::is_floating_point_v<FT>, "FT must be floating point");

    const DissimilarityMeasure measure_;
    MatrixType &data() const {return data_;}
    const VecT &row_sums() const {return row_sums_;}
    size_t size() const {return data_.rows();}
    template<typename PriorContainer=blaze::DynamicVector<FT, blaze::rowVector>>
    DissimilarityApplicator(MatrixType &ref,
                      DissimilarityMeasure measure=JSM,
                      Prior prior=NONE,
                      const PriorContainer *c=nullptr):
        data_(ref), measure_(measure)
    {
        prep(prior, c);
        MINOCORE_REQUIRE(dist::detail::is_valid_measure(measure_), "measure_ must be valid");
    }
    /*
     * Sets distance matrix, under measure_ (if not provided)
     * or measure (if provided as an argument).
     */
    template<typename MatType>
    void set_distance_matrix(MatType &m, bool symmetrize=false) const {set_distance_matrix(m, measure_, symmetrize);}

    template<typename MatType, DissimilarityMeasure measure>
    void set_distance_matrix(MatType &m, bool symmetrize=false) const {
        using blaze::sqrt;
        const size_t nr = m.rows();
        assert(nr == m.columns());
        assert(nr == data_.rows());
        static constexpr DissimilarityMeasure actual_measure =
            measure == JSM ? JSD
                : measure == COSINE_DISTANCE ? COSINE_SIMILARITY
                : measure == PROBABILITY_COSINE_DISTANCE ? PROBABILITY_COSINE_SIMILARITY
                : measure;
        for(size_t i = 0; i < nr; ++i) {
            if constexpr((blaze::IsDenseMatrix_v<MatrixType>)) {
                for(size_t j = i + 1; j < nr; ++j) {
                    m(i, j) = this->call<actual_measure>(i, j);
                }
            } else {
                OMP_PFOR
                for(size_t j = i + 1; j < nr; ++j) {
                    m(i, j) = this->call<actual_measure>(i, j);
                }
            }
        }
        if constexpr(measure == JSM) {
            if constexpr(blaze::IsDenseMatrix_v<MatType> || blaze::IsSparseMatrix_v<MatType>) {
                m = blaze::sqrt(m);
            } else if constexpr(dm::is_distance_matrix_v<MatType>) {
                blaze::CustomVector<FT, blaze::unaligned, blaze::unpadded> cv(const_cast<FT *>(m.data()), m.size());
                cv = blaze::sqrt(cv);
            } else {
                std::transform(m.begin(), m.end(), m.begin(), [](auto x) {return std::sqrt(x);});
            }
        } else if constexpr(measure == COSINE_DISTANCE || measure == PROBABILITY_COSINE_DISTANCE) {
            if constexpr(blaze::IsDenseMatrix_v<MatType> || blaze::IsSparseMatrix_v<MatType>) {
                m = blaze::acos(m) * PI_INV;
            } else if constexpr(dm::is_distance_matrix_v<MatType>) {
                blaze::CustomVector<FT, blaze::unaligned, blaze::unpadded> cv(const_cast<FT *>(m.data()), m.size());
                cv = blaze::acos(cv) * PI_INV;
            } else {
                std::transform(m.begin(), m.end(), m.begin(), [](auto x) {return std::acos(x) * PI_INV;});
            }
        }
        if constexpr(detail::is_symmetric(measure)) {
            if(symmetrize)
                fill_symmetric_upper_triangular(m);
        } else {
            if constexpr(dm::is_distance_matrix_v<MatType>) {
                std::fprintf(stderr, "Warning: using asymmetric measure with an upper triangular matrix. You are computing only half the values");
            } else {
                //std::fprintf(stderr, "Asymmetric measure %s/%s\n", detail::prob2str(measure), detail::prob2desc(measure));
                for(size_t i = 1; i < nr; ++i) {
                    if constexpr((blaze::IsDenseMatrix_v<MatrixType>)) {
                        //std::fprintf(stderr, "Filling bottom half\n");
                        for(size_t j = 0; j < i; ++j) {
                            auto v = this->call<measure>(i, j);
                            m(i, j) = v;
                        }
                    } else {
                        OMP_PFOR
                        for(size_t j = 0; j < i; ++j) {
                            auto v = this->call<measure>(i, j);
                            m(i, j) = v;
                        }
                    }
                    m(i, i) = 0.;
                }
            }
        }
    } // set_distance_matrix
    template<typename MatType>
    void set_distance_matrix(MatType &m, DissimilarityMeasure measure, bool symmetrize=false) const {
        switch(measure) {
            case TOTAL_VARIATION_DISTANCE: set_distance_matrix<MatType, TOTAL_VARIATION_DISTANCE>(m, symmetrize); break;
            case L1:                       set_distance_matrix<MatType, L1>(m, symmetrize); break;
            case L2:                       set_distance_matrix<MatType, L2>(m, symmetrize); break;
            case SQRL2:                    set_distance_matrix<MatType, SQRL2>(m, symmetrize); break;
            case PL2:                      set_distance_matrix<MatType, PL2>(m, symmetrize); break;
            case PSL2:                     set_distance_matrix<MatType, PSL2>(m, symmetrize); break;
            case JSD:                      set_distance_matrix<MatType, JSD>(m, symmetrize); break;
            case JSM:                      set_distance_matrix<MatType, JSM>(m, symmetrize); break;
            case REVERSE_MKL:              set_distance_matrix<MatType, REVERSE_MKL>(m, symmetrize); break;
            case MKL:                      set_distance_matrix<MatType, MKL>(m, symmetrize); break;
            case EMD:                      set_distance_matrix<MatType, EMD>(m, symmetrize); break;
            case WEMD:                     set_distance_matrix<MatType, WEMD>(m, symmetrize); break;
            case REVERSE_POISSON:          set_distance_matrix<MatType, REVERSE_POISSON>(m, symmetrize); break;
            case POISSON:                  set_distance_matrix<MatType, POISSON>(m, symmetrize); break;
            case HELLINGER:                set_distance_matrix<MatType, HELLINGER>(m, symmetrize); break;
            case BHATTACHARYYA_METRIC:     set_distance_matrix<MatType, BHATTACHARYYA_METRIC>(m, symmetrize); break;
            case BHATTACHARYYA_DISTANCE:   set_distance_matrix<MatType, BHATTACHARYYA_DISTANCE>(m, symmetrize); break;
            case LLR:                      set_distance_matrix<MatType, LLR>(m, symmetrize); break;
            case UWLLR:                    set_distance_matrix<MatType, UWLLR>(m, symmetrize); break;
            case OLLR:                     set_distance_matrix<MatType, OLLR>(m, symmetrize); break;
            case ITAKURA_SAITO:            set_distance_matrix<MatType, ITAKURA_SAITO>(m, symmetrize); break;
            case REVERSE_ITAKURA_SAITO:    set_distance_matrix<MatType, REVERSE_ITAKURA_SAITO>(m, symmetrize); break;
            case COSINE_DISTANCE:          set_distance_matrix<MatType, COSINE_DISTANCE>(m, symmetrize); break;
            case PROBABILITY_COSINE_DISTANCE:
                                           set_distance_matrix<MatType, PROBABILITY_COSINE_DISTANCE>(m, symmetrize); break;
            case COSINE_SIMILARITY:        set_distance_matrix<MatType, COSINE_SIMILARITY>(m, symmetrize); break;
            case PROBABILITY_COSINE_SIMILARITY:
                                           set_distance_matrix<MatType, PROBABILITY_COSINE_SIMILARITY>(m, symmetrize); break;
            case ORACLE_METRIC: case ORACLE_PSEUDOMETRIC: std::fprintf(stderr, "These are placeholders and should not be called."); throw std::invalid_argument("Placeholders");
            default: throw std::invalid_argument(std::string("unknown dissimilarity measure: ") + std::to_string(int(measure)) + dist::detail::prob2str(measure));
        }
    }
    template<typename OFT=FT>
    blaze::DynamicMatrix<OFT> make_distance_matrix(bool symmetrize=false) const {
        return make_distance_matrix<OFT>(measure_, symmetrize);
    }
    template<typename OFT=FT>
    blaze::DynamicMatrix<OFT> make_distance_matrix(DissimilarityMeasure measure, bool symmetrize=false) const {
        blaze::DynamicMatrix<OFT> ret(data_.rows(), data_.rows());
        set_distance_matrix(ret, measure, symmetrize);
        return ret;
    }
    FT cosine_similarity(size_t i, size_t j) const {
        return blaze::dot(weighted_row(i), weighted_row(j)) * l2norm_cache_->operator[](i) * l2norm_cache_->operator[](j);
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    FT cosine_similarity(size_t j, const OT &o) const {
        if constexpr(IS_SPARSE) {
            if(prior_data_) {
                if(prior_data_->size() != 1) throw TODOError("Disuniform prior");
                const auto pv = (*prior_data_)[0];
                if constexpr(blz::IsSparseVector_v<OT>) {
                    throw TODOError("Sparse-within but not without");
                } else {
                    auto v = blaze::dot(o + pv, weighted_row(j));
                    auto extra = blz::sum(o * pv);
                    auto rhn = blz::sqrt(blz::sum(blz::pow(o + pv, 2)));
                    return (v + extra) * (*l2norm_cache_)[j] / rhn;
                }
            }
        }
        return blaze::dot(o, weighted_row(j)) / blaze::l2Norm(o) * l2norm_cache_->operator[](j);
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    FT cosine_similarity(const OT &o, size_t j) const {
        return cosine_similarity(j, o);
    }
    FT pcosine_similarity(size_t i, size_t j) const {
        if(pl2norm_cache_) {
            return blaze::dot(row(i), row(j)) * pl2norm_cache_->operator[](i) * pl2norm_cache_->operator[](j);
        } else {
            return blaze::dot(row(i), row(j)) / (blz::l2Norm(row(i)) * blz::l2Norm(row(j)));
        }
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    FT pcosine_similarity(size_t j, const OT &o) const {
        if constexpr(IS_SPARSE) {
            if(prior_data_) {
#if 0
                if(prior_data_->size() != 1) throw TODOError("Disuniform prior");
                const auto pv = (*prior_data_)[0];
                const auto pvi = pv / (row_sums_[j] + prior_sum_);
                if constexpr(blz::IsSparseVector_v<OT>) {
                    throw TODOError("Sparse-within but not without");
                } else {
                    auto v = blaze::dot(o + pvi, row(j));
                    auto extra = blz::sum(o * pvi);
                    auto rhn = blz::sqrt(blz::sum(blz::pow(o + pvi, 2)));
                    return (v + extra) * (*l2norm_cache_)[j] / rhn;
                }
#else
                // dot(x + a, y + a)
                // ==
                // dot(x, y) + dot(x, a) + dot(y, a) + dot(a, a)
                if(prior_data_->size() == 1) {
                    const auto pv = (*prior_data_)[0];
                    auto rsj = row_sums_[j];
                    const auto pvi = 1. / (prior_sum_ + rsj);
                    double dim = data_.columns();
                    auto dotxy = blaze::dot(o, row(j));
                    //auto dotxa = (blaze::dot(o, [prior])) = blz::sum(o) * prior
                    auto dotxa = (rsj + prior_sum_);
                    auto osum = blz::sum(o);
                    auto dotya = osum + prior_sum_;
                    auto sum = (dotxa + dotya) * pv;
                    auto suma2 = data_.columns() * std::pow(pvi, 2);
                    auto num = dotxy + sum + suma2;
                    auto l2i = (*pl2norm_cache_)[j];
                    auto onorm = blz::sqrL2Norm(o);
                    // dot(o + a, o + a) = 
                    // dot(o, o) + 2 * dot(o, a) + dot(a, a)
                    auto onormsub = 2 * osum * pvi;
                    auto denom = std::sqrt(onorm + onormsub + suma2);
                    auto frac = num / (l2i * denom);
                    return frac;
                } else TODOError("fast sparse pcosine with disuniform prior");
#endif
            }
        }
        return blaze::dot(o, row(j)) / blaze::l2Norm(o) * pl2norm_cache_->operator[](j);
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    FT pcosine_similarity(const OT &o, size_t j) const {
        return pcosine_distance(j, o);
    }

    template<typename...Args>
    FT cosine_distance(Args &&...args) const {
        return std::acos(cosine_similarity(std::forward<Args>(args)...)) * PI_INV;
    }
    template<typename...Args>
    FT pcosine_distance(Args &&...args) const {
        return std::acos(pcosine_similarity(std::forward<Args>(args)...)) * PI_INV;
    }
    FT dotproduct_distance(size_t i, size_t j) const {
        return blaze::dot(weighted_row(i), weighted_row(j)) * l2norm_cache_->operator[](i) * l2norm_cache_->operator[](j);
    }
    FT pdotproduct_distance(size_t i, size_t j) const {
        return blaze::dot(row(i), row(j)) * pl2norm_cache_->operator[](i) * pl2norm_cache_->operator[](j);
    }


    // Accessors
    decltype(auto) weighted_row(size_t ind) const {
        return blaze::row(data_, ind BLAZE_CHECK_DEBUG) * row_sums_[ind];
    }
    decltype(auto) row(size_t ind) const {return blaze::row(data_, ind BLAZE_CHECK_DEBUG);}
    decltype(auto) logrow(size_t ind) const {return blaze::row(logdata_, ind BLAZE_CHECK_DEBUG);}
    decltype(auto) sqrtrow(size_t ind) const {return blaze::row(sqrdata_, ind BLAZE_CHECK_DEBUG);}

    /*
     * Distances
     */
    INLINE auto operator()(size_t i, size_t j) const {
        return this->operator()(i, j, measure_);
    }
    template<DissimilarityMeasure constexpr_measure, typename OT, typename CacheT,
             typename=std::enable_if_t<!std::is_integral_v<OT> && !std::is_integral_v<CacheT>>>
    INLINE FT operator()(size_t i, OT &o, CacheT *cp=static_cast<CacheT *>(nullptr)) const {
        return this->call<constexpr_measure, OT, CacheT>(i, o, cp);
    }
    template<DissimilarityMeasure constexpr_measure, typename OT, typename CacheT,
             typename=std::enable_if_t<!std::is_integral_v<OT> && !std::is_integral_v<CacheT>>>
    INLINE FT operator()(OT &o, size_t i, CacheT *cp=static_cast<CacheT *>(nullptr)) const {
        return this->call<constexpr_measure, OT, CacheT>(o, i, cp);
    }
    template<DissimilarityMeasure constexpr_measure, typename OT, typename CacheT=OT,
             typename=std::enable_if_t<!std::is_integral_v<OT> && !std::is_integral_v<CacheT>>>
    INLINE FT call(OT &o, size_t i, CacheT *cp=static_cast<CacheT *>(nullptr)) const {
        FT ret;
        if constexpr(constexpr_measure == TOTAL_VARIATION_DISTANCE) {
            ret = discrete_total_variation_distance(o, row(i));
        } else if constexpr(constexpr_measure == L1) {
            ret = l1Norm(weighted_row(i) - o);
        } else if constexpr(constexpr_measure == L2) {
            ret = l2Norm(weighted_row(i) - o);
        } else if constexpr(constexpr_measure == SQRL2) {
            ret = blaze::sqrNorm(weighted_row(i) - o);
        } else if constexpr(constexpr_measure == PSL2) {
            ret = blaze::sqrNorm(i - o);
        } else if constexpr(constexpr_measure == PL2) {
            ret = p_l2norm(i, o);
        } else if constexpr(constexpr_measure == JSD) {
            if(cp) {
                ret = jsd(i, o, *cp);
            } else ret = jsd(i, o);
        } else if constexpr(constexpr_measure == JSM) {
            if(cp) {
                ret = jsm(i, o, *cp);
            } else ret = jsm(i, o);
        } else if constexpr(constexpr_measure == REVERSE_MKL) {
            ret = cp ? mkl(i, o, *cp): mkl(i, o);
        } else if constexpr(constexpr_measure == MKL) {
            ret = cp ? mkl(o, i, *cp): mkl(o, i);
        } else if constexpr(constexpr_measure == EMD) {
            throw NotImplementedError("Removed: currently, p_wasserstein functions are not permitted");
            ret = p_wasserstein(row(i), o);
        } else if constexpr(constexpr_measure == WEMD) {
            throw NotImplementedError("Removed: currently, p_wasserstein functions are not permitted");
            ret = p_wasserstein(weighted_row(i), o);
        } else if constexpr(constexpr_measure == REVERSE_POISSON) {
            ret = cp ? pkl(i, o, *cp): pkl(i, o);
        } else if constexpr(constexpr_measure == POISSON) {
            ret = cp ? pkl(o, i, *cp): pkl(o, i);
        } else if constexpr(constexpr_measure == HELLINGER) {
            ret = cp ? blaze::sqrNorm(sqrtrow(i) - *cp)
                     : blaze::sqrNorm(sqrtrow(i) - blaze::sqrt(o));
        } else if constexpr(constexpr_measure == BHATTACHARYYA_METRIC) {
            ret = bhattacharyya_metric(i, o);
        } else if constexpr(constexpr_measure == BHATTACHARYYA_DISTANCE) {
            ret = bhattacharyya_distance(i, o);
        } else if constexpr(constexpr_measure == LLR) {
            ret = cp ? llr(i, o, *cp): llr(i, o);
        } else if constexpr(constexpr_measure == UWLLR) {
            ret = cp ? uwllr(i, o, *cp): uwllr(i, o);
        } else if constexpr(constexpr_measure == OLLR) {
            throw 1; // Not implemented
        } else if constexpr(constexpr_measure == ITAKURA_SAITO) {
            ret = itakura_saito(o, i);
        } else if constexpr(constexpr_measure == REVERSE_ITAKURA_SAITO) {
            ret = itakura_saito(i, o);
        } else if constexpr(constexpr_measure == COSINE_DISTANCE) {
            ret = cosine_distance(i, o);
        } else if constexpr(constexpr_measure == PROBABILITY_COSINE_DISTANCE) {
            ret = pcosine_distance(i, o);
        } else if constexpr(constexpr_measure == COSINE_SIMILARITY) {
            ret = cosine_similarity(i, o);
        } else if constexpr(constexpr_measure == PROBABILITY_COSINE_SIMILARITY) {
            ret = pcosine_similarity(i, o);
        } else {
            throw std::runtime_error(std::string("Unknown measure: ") + std::to_string(int(constexpr_measure)));
        }
        return ret;
    }
    template<DissimilarityMeasure constexpr_measure, typename OT, typename CacheT=OT,
             typename=std::enable_if_t<!std::is_integral_v<OT> && !std::is_integral_v<CacheT>>>
    INLINE FT call(size_t i, OT &o, [[maybe_unused]] CacheT *cp=static_cast<CacheT *>(nullptr)) const {
        FT ret;
        assert(i < this->data().rows());
        if constexpr(constexpr_measure == TOTAL_VARIATION_DISTANCE) {
            ret = discrete_total_variation_distance(row(i), o);
        } else if constexpr(constexpr_measure == L1) {
            ret = l1Norm(weighted_row(i) - o);
        } else if constexpr(constexpr_measure == L2) {
            ret = l2Norm(weighted_row(i) - o);
        } else if constexpr(constexpr_measure == PL2) {
            ret = p_l2norm(i, o);
        } else if constexpr(constexpr_measure == SQRL2) {
            ret = blaze::sqrNorm(weighted_row(i) - o);
            //std::fprintf(stderr, "SQRL2 between row %zu and row starting at %p is %g\n", i, (void *)&*o.begin(), ret);
        } else if constexpr(constexpr_measure == PSL2) {
            ret = blaze::sqrNorm(i - o);
        } else if constexpr(constexpr_measure == JSD) {
            if(cp) {
                ret = jsd(i, o, *cp);
            } else ret = jsd(i, o);
        } else if constexpr(constexpr_measure == JSM) {
            if(cp) {
                ret = jsm(i, o, *cp);
            } else ret = jsm(i, o);
        } else if constexpr(constexpr_measure == REVERSE_MKL) {
            if(cp) {
                ret = mkl(o, i, *cp);
            } else ret = mkl(o, i);
        } else if constexpr(constexpr_measure == MKL) {
            if(cp) {
                ret = mkl(i, o, *cp);
            } else ret = mkl(i, o);
        } else if constexpr(constexpr_measure == EMD) {
            throw NotImplementedError("Removed: currently, p_wasserstein functions are not permitted");
            ret = p_wasserstein(row(i), o);
        } else if constexpr(constexpr_measure == WEMD) {
            throw NotImplementedError("Removed: currently, p_wasserstein functions are not permitted");
            ret = p_wasserstein(weighted_row(i), o);
        } else if constexpr(constexpr_measure == REVERSE_POISSON) {
            ret = cp ? pkl(o, i, *cp): pkl(o, i);
        } else if constexpr(constexpr_measure == POISSON) {
            ret = cp ? pkl(i, o, *cp): pkl(i, o);
        } else if constexpr(constexpr_measure == HELLINGER) {
            if(cp) {
                ret = blaze::sqrNorm(sqrtrow(i) - *cp);
            } else {
                ret = blaze::sqrNorm(sqrtrow(i) - blaze::sqrt(o));
            }
        } else if constexpr(constexpr_measure == BHATTACHARYYA_METRIC) {
            ret = cp ? bhattacharyya_metric(i, o, *cp)
                     : bhattacharyya_metric(i, o);
        } else if constexpr(constexpr_measure == BHATTACHARYYA_DISTANCE) {
            ret = cp ? bhattacharyya_distance(i, o, *cp)
                     : bhattacharyya_distance(i, o);
        } else if constexpr(constexpr_measure == LLR) {
            ret = cp ? llr(i, o, *cp): llr(i, o);
        } else if constexpr(constexpr_measure == UWLLR) {
            ret = cp ? uwllr(i, o, *cp): uwllr(i, o);
        } else if constexpr(constexpr_measure == OLLR) {
            ret = cp ? llr(i, o, *cp): llr(i, o);
            std::cerr << "Note: computing LLR, not OLLR, for this case\n";
        } else if constexpr(constexpr_measure == ITAKURA_SAITO) {
            ret = itakura_saito(i, o);
        } else if constexpr(constexpr_measure == REVERSE_ITAKURA_SAITO) {
            ret = itakura_saito(o, i);
        } else if constexpr(constexpr_measure == COSINE_DISTANCE) {
            ret = cosine_distance(i, o);
        } else if constexpr(constexpr_measure == PROBABILITY_COSINE_DISTANCE) {
            ret = pcosine_distance(i, o);
        } else if constexpr(constexpr_measure == COSINE_SIMILARITY) {
            ret = cosine_similarity(i, o);
        } else if constexpr(constexpr_measure == PROBABILITY_COSINE_SIMILARITY) {
            ret = pcosine_similarity(i, o);
        } else {
            throw std::runtime_error(std::string("Unknown measure: ") + std::to_string(int(constexpr_measure)));
        }
        return ret;
    }
    template<DissimilarityMeasure constexpr_measure>
    INLINE FT call(size_t i, size_t j) const {
        FT ret;
        if constexpr(constexpr_measure == TOTAL_VARIATION_DISTANCE) {
            ret = discrete_total_variation_distance(row(i), row(j));
        } else if constexpr(constexpr_measure == L1) {
            ret = l1Norm(weighted_row(i) - weighted_row(j));
        } else if constexpr(constexpr_measure == L2) {
            ret = l2Norm(weighted_row(i) - weighted_row(j));
        } else if constexpr(constexpr_measure == PL2) {
            ret = p_l2norm(i, j);
        } else if constexpr(constexpr_measure == SQRL2) {
            ret = blaze::sqrNorm(weighted_row(i) - weighted_row(j));
        } else if constexpr(constexpr_measure == PSL2) {
            ret = blaze::sqrNorm(row(i) - row(j));
        } else if constexpr(constexpr_measure == JSD) {
            ret = jsd(i, j);
        } else if constexpr(constexpr_measure == JSM) {
            ret = jsm(i, j);
        } else if constexpr(constexpr_measure == REVERSE_MKL) {
            ret = mkl(j, i);
        } else if constexpr(constexpr_measure == MKL) {
            ret = mkl(i, j);
        } else if constexpr(constexpr_measure == EMD) {
            throw NotImplementedError("Removed: currently, p_wasserstein functions are not permitted");
            ret = p_wasserstein(row(i), row(j));
        } else if constexpr(constexpr_measure == WEMD) {
            throw NotImplementedError("Removed: currently, p_wasserstein functions are not permitted");
            ret = p_wasserstein(weighted_row(i), weighted_row(j));
        } else if constexpr(constexpr_measure == REVERSE_POISSON) {
            ret = pkl(j, i);
        } else if constexpr(constexpr_measure == POISSON) {
            ret = pkl(i, j);
        } else if constexpr(constexpr_measure == HELLINGER) {
            ret = hellinger(i, j);
        } else if constexpr(constexpr_measure == BHATTACHARYYA_METRIC) {
            ret = bhattacharyya_metric(i, j);
        } else if constexpr(constexpr_measure == BHATTACHARYYA_DISTANCE) {
            ret = bhattacharyya_distance(i, j);
        } else if constexpr(constexpr_measure == LLR) {
            ret = llr(i, j);
        } else if constexpr(constexpr_measure == UWLLR) {
            ret = uwllr(i, j);
        } else if constexpr(constexpr_measure == OLLR) {
            ret = ollr(i, j);
        } else if constexpr(constexpr_measure == ITAKURA_SAITO) {
            ret = itakura_saito(i, j);
        } else if constexpr(constexpr_measure == REVERSE_ITAKURA_SAITO) {
            ret = itakura_saito(j, i);
        } else if constexpr(constexpr_measure == COSINE_DISTANCE) {
            ret = cosine_distance(j, i);
        } else if constexpr(constexpr_measure == PROBABILITY_COSINE_DISTANCE) {
            ret = pcosine_distance(j, i);
        } else if constexpr(constexpr_measure == COSINE_SIMILARITY) {
            ret = cosine_similarity(j, i);
        } else if constexpr(constexpr_measure == PROBABILITY_COSINE_SIMILARITY) {
            ret = pcosine_similarity(j, i);
        } else {
            throw std::runtime_error(std::string("Unknown measure: ") + std::to_string(int(constexpr_measure)));
        }
        return ret;
    }
    template<typename OT, typename CacheT=OT, typename=std::enable_if_t<!std::is_integral_v<OT> > >
    INLINE FT operator()(const OT &o, size_t i, const CacheT *cache=static_cast<CacheT *>(nullptr)) const noexcept {
        return this->operator()(o, i, cache, measure_);
    }
    template<typename OT, typename CacheT=OT, typename=std::enable_if_t<!std::is_integral_v<OT> > >
    INLINE FT operator()(const OT &o, size_t i, const CacheT *cache, DissimilarityMeasure measure) const noexcept {
        if(unlikely(measure == static_cast<DissimilarityMeasure>(-1))) {
            std::cerr << "Unset measure\n";
            std::exit(1);
        }
        FT ret;
        switch(measure) {
            case TOTAL_VARIATION_DISTANCE: ret = call<TOTAL_VARIATION_DISTANCE>(o, i); break;
            case L1: ret = call<L1>(o, i); break;
            case L2: ret = call<L2>(o, i); break;
            case SQRL2: ret = call<SQRL2>(o, i); break;
            case PSL2: ret = call<PSL2>(o, i); break;
            case PL2: ret = call<PL2>(o, i); break;
            case JSD: ret = call<JSD>(o, i); break;
            case JSM: ret = call<JSM>(o, i); break;
            case REVERSE_MKL: ret = call<REVERSE_MKL>(o, i, cache); break;
            case MKL: ret = call<MKL>(o, i, cache); break;
            case EMD: ret = call<EMD>(o, i); break;
            case WEMD: ret = call<WEMD>(o, i); break;
            case REVERSE_POISSON: ret = call<REVERSE_POISSON>(o, i, cache); break;
            case POISSON: ret = call<POISSON>(o, i, cache); break;
            case HELLINGER: ret = call<HELLINGER>(o, i, cache); break;
            case BHATTACHARYYA_METRIC: ret = call<BHATTACHARYYA_METRIC>(o, i); break;
            case BHATTACHARYYA_DISTANCE: ret = call<BHATTACHARYYA_DISTANCE>(o, i); break;
            case LLR: ret = call<LLR>(o, i, cache); break;
            case UWLLR: ret = call<UWLLR>(o, i, cache); break;
            case OLLR: ret = call<OLLR>(o, i, cache); break;
            case ITAKURA_SAITO: ret = call<ITAKURA_SAITO>(o, i, cache); break;
            case COSINE_DISTANCE: ret = call<COSINE_DISTANCE>(o, i); break;
            case PROBABILITY_COSINE_DISTANCE: ret = call<PROBABILITY_COSINE_DISTANCE>(o, i); break;
            case COSINE_SIMILARITY: ret = call<COSINE_SIMILARITY>(o, i); break;
            case PROBABILITY_COSINE_SIMILARITY: ret = call<PROBABILITY_COSINE_SIMILARITY>(o, i); break;
            case ORACLE_METRIC: case ORACLE_PSEUDOMETRIC: std::fprintf(stderr, "These are placeholders and should not be called."); return 0.;
            default: __builtin_unreachable();
        }
        return ret;
    }
    template<typename OT, typename CacheT=OT, typename=std::enable_if_t<!std::is_integral_v<OT> > >
    INLINE FT operator()(size_t i, const OT &o, const CacheT *cache=static_cast<CacheT *>(nullptr)) const {
        return this->operator()(i, o, cache, measure_);
    }
    template<typename OT, typename CacheT=OT, typename=std::enable_if_t<!std::is_integral_v<OT> > >
    INLINE FT operator()(size_t i, const OT &o, const CacheT *cache, DissimilarityMeasure measure) const noexcept {
        if(unlikely(i >= data_.rows())) {
            std::cerr << (std::string("Invalid rows selection: ") + std::to_string(i) + '\n');
            std::exit(1);
        }
        if(unlikely(measure == static_cast<DissimilarityMeasure>(-1))) {
            std::cerr << "Unset measure\n";
            std::exit(1);
        }
        FT ret;
        switch(measure) {
            case TOTAL_VARIATION_DISTANCE: ret = call<TOTAL_VARIATION_DISTANCE>(i, o); break;
            case L1: ret = call<L1>(i, o); break;
            case L2: ret = call<L2>(i, o); break;
            case PL2: ret = call<PL2>(i, o); break;
            case PSL2: ret = call<PSL2>(i, o); break;
            case SQRL2: ret = call<SQRL2>(i, o); break;
            case JSD: ret = call<JSD>(i, o); break;
            case JSM: ret = call<JSM>(i, o); break;
            case REVERSE_MKL: ret = call<REVERSE_MKL>(i, o, cache); break;
            case MKL: ret = call<MKL>(i, o, cache); break;
            case EMD: ret = call<EMD>(i, o); break;
            case WEMD: ret = call<WEMD>(i, o); break;
            case REVERSE_POISSON: ret = call<REVERSE_POISSON>(i, o, cache); break;
            case POISSON: ret = call<POISSON>(i, o, cache); break;
            case HELLINGER: ret = call<HELLINGER>(i, o, cache); break;
            case BHATTACHARYYA_METRIC: ret = call<BHATTACHARYYA_METRIC>(i, o); break;
            case BHATTACHARYYA_DISTANCE: ret = call<BHATTACHARYYA_DISTANCE>(i, o); break;
            case LLR: ret = call<LLR>(i, o, cache); break;
            case UWLLR: ret = call<UWLLR>(i, o, cache); break;
            case OLLR: ret = call<OLLR>(i, o, cache); break;
            case ITAKURA_SAITO: ret = call<ITAKURA_SAITO>(i, o, cache); break;
            case COSINE_DISTANCE: ret = call<COSINE_DISTANCE>(i, o); break;
            case PROBABILITY_COSINE_DISTANCE: ret = call<PROBABILITY_COSINE_DISTANCE>(i, o); break;
            case COSINE_SIMILARITY: ret = call<COSINE_SIMILARITY>(i, o); break;
            case PROBABILITY_COSINE_SIMILARITY: ret = call<PROBABILITY_COSINE_SIMILARITY>(i, o); break;
            case ORACLE_METRIC: case ORACLE_PSEUDOMETRIC: std::fprintf(stderr, "These are placeholders and should not be called."); return 0.;
            default: __builtin_unreachable();
        }
        return ret;
    }
    INLINE FT operator()(size_t i, size_t j, DissimilarityMeasure measure) const noexcept {
        if(unlikely(i >= data_.rows() || j >= data_.rows())) {
            std::cerr << (std::string("Invalid rows selection: ") + std::to_string(i) + ", " + std::to_string(j) + '\n');
            std::exit(1);
        }
        FT ret;
        switch(measure) {
            case TOTAL_VARIATION_DISTANCE: ret = call<TOTAL_VARIATION_DISTANCE>(i, j); break;
            case L1: ret = call<L1>(i, j); break;
            case L2: ret = call<L2>(i, j); break;
            case PL2: ret = call<PL2>(i, j); break;
            case PSL2: ret = call<PSL2>(i, j); break;
            case SQRL2: ret = call<SQRL2>(i, j); break;
            case JSD: ret = call<JSD>(i, j); break;
            case JSM: ret = call<JSM>(i, j); break;
            case REVERSE_MKL: ret = call<REVERSE_MKL>(i, j); break;
            case MKL: ret = call<MKL>(i, j); break;
            case EMD: ret = call<EMD>(i, j); break;
            case WEMD: ret = call<WEMD>(i, j); break;
            case REVERSE_POISSON: ret = call<REVERSE_POISSON>(i, j); break;
            case POISSON: ret = call<POISSON>(i, j); break;
            case HELLINGER: ret = call<HELLINGER>(i, j); break;
            case BHATTACHARYYA_METRIC: ret = call<BHATTACHARYYA_METRIC>(i, j); break;
            case BHATTACHARYYA_DISTANCE: ret = call<BHATTACHARYYA_DISTANCE>(i, j); break;
            case LLR: ret = call<LLR>(i, j); break;
            case UWLLR: ret = call<UWLLR>(i, j); break;
            case OLLR: ret = call<OLLR>(i, j); break;
            case ITAKURA_SAITO: ret = call<ITAKURA_SAITO>(i, j); break;
            case COSINE_DISTANCE: ret = call<COSINE_DISTANCE>(i, j); break;
            case PROBABILITY_COSINE_DISTANCE: ret = call<PROBABILITY_COSINE_DISTANCE>(i, j); break;
            case COSINE_SIMILARITY: ret = call<COSINE_SIMILARITY>(i, j); break;
            case PROBABILITY_COSINE_SIMILARITY: ret = call<PROBABILITY_COSINE_SIMILARITY>(i, j); break;
            case ORACLE_METRIC: case ORACLE_PSEUDOMETRIC: std::fprintf(stderr, "These are placeholders and should not be called."); return 0.;
            default: __builtin_unreachable();
        }
        return ret;
    }
    template<typename MatType>
    void operator()(MatType &mat, DissimilarityMeasure measure, bool symmetrize=false) {
        set_distance_matrix(mat, measure, symmetrize);
    }
    template<typename MatType, typename=std::enable_if_t<!std::is_arithmetic_v<MatType>>>
    void operator()(MatType &mat, bool symmetrize=false) {
        set_distance_matrix(mat, symmetrize);
    }
    template<typename MatType>
    auto operator()() {
        return make_distance_matrix(measure_);
    }

#if 0
    template<typename F1, typename F2, typename F3, typename F4>
    void generic_sparse_uniform_prior(size_t i, size_t j, const F1 &fshare, const F2 &lho, const F3 &rho, const F4 &sharedz) const {
        size_t ci = 0;
        auto lhr = row(i), rhr = row(j);
        auto lhit = lhr.begin(), lhe = lhr.end(), rhit = rhr.begin(), rhe = rhr.end();
        unsigned shared_zeros = 0;
        size_t nextindex;
        for(;;) {
            switch(((lhit == lhe) << 1) | (rhit == rhe)) {
                case 0: {
                    auto lhi = lhit->index(), rhi = rhit->index();
                    nextindex = std::min(lhi, rhi);
                }
            }
        }
    }
#endif

    FT itakura_saito(size_t i, size_t j) const {
        FT ret;
        if constexpr(IS_SPARSE) {
            if(!prior_data_) {
                char buf[128];
                std::sprintf(buf, "warning: Itakura-Saito cannot be computed to sparse vectors/matrices at %zu/%zu\n", i, j);
                throw std::runtime_error(buf);
            }
            auto do_inc = [&](auto x) ALWAYS_INLINE {ret += x - std::log(x);};
            const size_t dim = data_.columns();
            auto lhn = row_sums_[i] + prior_sum_, rhn = row_sums_[j] + prior_sum_;
            auto lhi = 1. / lhn, rhi = 1. / rhn;
            auto lhr(row(i)), rhr(row(j));
            if(prior_data_->size() == 1) {
                const auto mul = prior_data_->operator[](0);
                const auto lhrsi = mul / lhn, rhrsi = mul / rhn;
                ret = -static_cast<FT>(dim); // To account for -1 in IS distance.
                const auto shared_zero = merge::for_each_by_case(dim, lhr.begin(), lhr.end(), rhr.begin(), rhr.end(),
                    [&](auto, auto x, auto y) ALWAYS_INLINE {do_inc((x + lhrsi) / (y + rhrsi));},
                    [&](auto, auto x) ALWAYS_INLINE {do_inc((x + lhrsi) * rhn);},
                    [&](auto, auto y) ALWAYS_INLINE {do_inc(lhrsi / (y + rhrsi));});
                const auto lhrsirhnp = lhrsi * rhn;
                ret += shared_zero * (lhrsirhnp - std::log(lhrsirhnp)); // Account for shared-zero positions
            } else {
                auto &pd(*prior_data_);
                merge::for_each_by_case(dim, lhr.begin(), lhr.end(), rhr.begin(), rhr.end(),
                    [&](auto idx, auto x, auto y) ALWAYS_INLINE {do_inc((x + pd[idx] * lhi) / (y + pd[idx] * rhi));},
                    [&](auto idx, auto x) ALWAYS_INLINE {do_inc((x + pd[idx] * lhi) / (pd[idx] * rhi));},
                    [&](auto idx, auto y) ALWAYS_INLINE {do_inc(pd[idx] * lhi / (y + pd[idx] * rhi));});
            }
        } else {
            auto div = blaze::evaluate(row(i) / row(j));
            ret = blaze::sum(div - blaze::log(div)) - row(i).size();
        }
        return ret;
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>> >
    FT itakura_saito(size_t i, const OT &o) const {
        FT ret;
        if constexpr(IS_SPARSE) {
            if(!prior_data_) {
                char buf[128];
                std::sprintf(buf, "warning: Itakura-Saito cannot be computed to sparse vectors/matrices at %zu/%p\n", i, (void *)&o);
                throw std::runtime_error(buf);
            }
            auto do_inc = [&](auto x) ALWAYS_INLINE {ret += x - std::log(x);};
            const size_t dim = data_.columns();
            auto lhn = row_sums_[i] + prior_sum_, rhn = blz::sum(o) + prior_sum_;
            auto lhi = 1. / lhn, rhi = 1. / rhn;
            auto lhr(row(i));
            ret = -static_cast<FT>(dim); // To account for -1 in IS distance.
            if(prior_data_->size() == 1) {
                const auto mul = prior_data_->operator[](0);
                const auto lhrsi = mul / lhn, rhrsi = mul / rhn;
                const auto shared_zero = merge::for_each_by_case(dim, lhr.begin(), lhr.end(), o.begin(), o.end(),
                    [&](auto, auto x, auto y) ALWAYS_INLINE {do_inc((x + lhrsi) / (y + rhrsi));},
                    [&](auto, auto x) ALWAYS_INLINE {do_inc((x + lhrsi) * rhn);},
                    [&](auto, auto y) ALWAYS_INLINE {do_inc(lhrsi / (y + rhrsi));});
                const auto lhrsirhnp = lhrsi * rhn;
                ret += shared_zero * (lhrsirhnp - std::log(lhrsirhnp)); // Account for shared-zero positions
            } else {
                auto &pd(*prior_data_);
                merge::for_each_by_case(dim, lhr.begin(), lhr.end(), o.begin(), o.end(),
                    [&](auto idx, auto x, auto y) ALWAYS_INLINE {do_inc((x + pd[idx] * lhi) / (y + pd[idx] * rhi));},
                    [&](auto idx, auto x) ALWAYS_INLINE {do_inc((x + pd[idx] * lhi) / (pd[idx] * rhi));},
                    [&](auto idx, auto y) ALWAYS_INLINE {do_inc(pd[idx] * lhi / (y + pd[idx] * rhi));},
                    [&](auto idx) ALWAYS_INLINE {do_inc(lhi * rhn);});
            }
        } else {
            auto div = blaze::evaluate(row(i) / o);
            ret = blaze::sum(div - blaze::log(o)) - row(i).size();
        }
        return ret;
    }

    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>> >
    FT p_l2norm(size_t i, const OT &o) const {
#ifdef VERBOSE_AF
        if constexpr(IS_SPARSE) {
            if(prior_data_ && !warning_emitted) {
                warning_emitted = true;
                std::fprintf(stderr, "Note: p_l2norm with a prior is not specialized.\n");
            }
        }
#endif
        return blz::l2Norm(row(i) - o);
    }
    FT p_l2norm(size_t i, size_t j) const {
#ifdef VERBOSE_AF
        if constexpr(IS_SPARSE) {
            if(prior_data_ && !warning_emitted) {
                warning_emitted = true;
                std::fprintf(stderr, "Note: p_l2norm with a prior is not specialized.\n");
            }
        }
#endif
        return blz::l2Norm(row(i) - row(j));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>> >
    FT itakura_saito(const OT &o, size_t i) const {
        FT ret;
        if constexpr(IS_SPARSE) {
            if(!prior_data_) {
                char buf[128];
                std::sprintf(buf, "warning: Itakura-Saito cannot be computed to sparse vectors/matrices at %zu/%p\n", i, (void *)&o);
                throw std::runtime_error(buf);
            }
            ret = -std::numeric_limits<FT>::max();
            throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        } else {
            auto div = o / row(i);
            ret = blaze::sum(div - blaze::log(div)) - o.size();
        }
        return ret;
    }

    FT hellinger(size_t i, size_t j) const {
        return sqrdata_.rows() ? blaze::sqrNorm(sqrtrow(i) - sqrtrow(j))
                               : blaze::sqrNorm(blaze::sqrt(row(i)) - blaze::sqrt(row(j)));
    }
    FT jsd(size_t i, size_t j) const {
        FT ret;
        if(!IsSparseMatrix_v<MatrixType> || !prior_data_) {
            assert(i < data_.rows());
            assert(j < data_.rows());
            auto ri = row(i), rj = row(j);
            //constexpr FT logp5 = -0.693147180559945; // std::log(0.5)
            auto s = evaluate(ri + rj);
            ret = __getjsc(i) + __getjsc(j) - blaze::dot(s, blaze::neginf2zero(blaze::log(s * 0.5)));
            return std::max(static_cast<FT>(.5) * ret, static_cast<FT>(0.));
        }
        if constexpr(IS_SPARSE) {
            ret = __getjsc(i) + __getjsc(j);
            const size_t dim = data_.columns();
            auto lhr = row(i), rhr = row(j);
            auto lhit = lhr.begin(), rhit = rhr.begin();
            const auto lhe = lhr.end(), rhe = rhr.end();
            auto lhrsi = 1. / (row_sums_[i] + prior_sum_);
            auto rhrsi = 1. / (row_sums_[j] + prior_sum_);
            auto bothsi = lhrsi + rhrsi;
            if(prior_data_->size() == 1) {
                const auto lhrsimul = lhrsi * prior_data_->operator[](0);
                const auto rhrsimul = rhrsi * prior_data_->operator[](0);
                const auto bothsimul = lhrsimul + rhrsimul;
                auto dox = [&,bothsimul](auto x) ALWAYS_INLINE {
                    x += bothsimul;
                    ret -= x * std::log(.5 * x);
                };
                auto single_func = [&](auto, auto lhs) ALWAYS_INLINE {dox(lhs);};
                auto shared_zeros = merge::for_each_by_case(dim, lhit, lhe, rhit, rhe,
                    [&](auto, auto lhs, auto rhs) ALWAYS_INLINE {dox(lhs + rhs);},
                    single_func, single_func);
                ret -= shared_zeros * (bothsimul * std::log(.5 * (bothsimul)));
            } else {
                // This could later be accelerated, but that kind of caching is more complicated.
                const auto &pd = *prior_data_;
                auto dox = [&](auto x) ALWAYS_INLINE {ret -= x * std::log(.5 * x);};
                auto single_func = [&](auto idx, auto val) {dox(val + pd[idx] * bothsi);};
                merge::for_each_by_case(dim, lhit, lhe, rhit, rhe,
                    [&](auto idx, auto lhs, auto rhs) {dox(lhs + rhs + pd[idx] * bothsi);},
                    single_func, single_func,
                    [&](auto idx) {dox(pd[idx] * bothsi);}
                );
            }
            return ret * static_cast<FT>(.5);
        }
        __builtin_unreachable();
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    FT jsd(size_t i, const OT &o, const OT2 &olog) const {
        if(IS_SPARSE && blaze::IsSparseVector_v<OT> && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        auto mnlog = evaluate(log(0.5 * (row(i) + o)));
        return (blaze::dot(row(i), logrow(i) - mnlog) + blaze::dot(o, olog - mnlog));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    FT jsd(size_t i, const OT &o) const {
        if(IS_SPARSE && blaze::IsSparseVector_v<OT> && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        auto olog = evaluate(blaze::neginf2zero(blaze::log(o)));
        return jsd(i, o, olog);
    }
    FT mkl(size_t i, size_t j) const {
        if constexpr(IS_SPARSE) {
            if(prior_data_) {
                const auto &pd(*prior_data_);
                const bool single_value = pd.size() == 1;
                auto lhr = row(i);
                const size_t dim = lhr.size();
                auto rhr = row(j);
                auto lhit = lhr.begin(), rhit = rhr.begin();
                const auto lhe = lhr.end(), rhe = rhr.end();
                const auto lhrsi = 1. / (row_sums_[i] + prior_sum_);
                const auto rhrsi = 1. / (row_sums_[j] + prior_sum_);
                FT ret = 0.;
                if(single_value) {
                    const FT lhinc = pd[0] * lhrsi;
                    const FT rhinc = pd[0] * rhrsi;
                    const FT rhincl = std::log(rhinc);
                    const FT empty_contrib = -lhinc * rhincl;
                    auto nz = merge::for_each_by_case(dim, lhit, lhe, rhit, rhe,
                        [&](auto, auto xval, auto yval) ALWAYS_INLINE {ret -= (xval + lhinc) + std::log(yval + rhinc);},
                        [&](auto, auto xval) ALWAYS_INLINE  {ret -= (xval + lhinc) * rhincl;},
                        [&](auto, auto yval) ALWAYS_INLINE  {ret -= lhinc * std::log(yval + rhinc);});
                    ret += empty_contrib * nz;
                } else { // if(single_value) / else
                    merge::for_each_by_case(dim, lhit, lhe, rhit, rhe,
                        [&](auto idx, auto xval, auto yval) ALWAYS_INLINE {ret -= (xval + lhrsi * pd[idx]) * std::log(yval + rhrsi * pd[idx]);},
                        [&](auto idx, auto xval) ALWAYS_INLINE {ret -= (xval + lhrsi * pd[idx]) * std::log(rhrsi * pd[idx]);},
                        [&](auto idx, auto yval) ALWAYS_INLINE {ret -= lhrsi * pd[idx] * std::log(yval + rhrsi * pd[idx]);},
                        [&](auto idx) ALWAYS_INLINE {ret -= lhrsi * pd[idx] * std::log(rhrsi * pd[idx]);});
                }
                return ret + __getjsc(i);
            }
        }
        return FT(__getjsc(i) - blz::dot(row(i), logrow(j)));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    FT mkl(size_t i, const OT &o) const {
        FT ret = __getjsc(i);
        if constexpr(IS_SPARSE) {
            if(prior_data_) {
                const auto osum = blz::sum(o);
                auto &pd = *prior_data_;
                if(pd.size() == 1) {
                    auto r = row(i);
                    const size_t dim = r.size();
                    auto rb = r.begin(), re = r.end();
                    auto rs = row_sums_[i];
                    auto rsa = rs + prior_sum_;
                    auto orsa = osum + prior_sum_;
                    auto rsai = 1. / rsa;
                    auto orsai = 1. / orsa;
                    const auto pv = pd[0];
                    auto rsaimul = rsai * pv;
                    auto orsaimul = orsai * pv;
                    auto lorsaimul = std::log(orsaimul);
                    FT ret = 0;
                    size_t ind = 0;
                    if constexpr(blz::IsSparseVector_v<OT>) {
                        auto sharedz = merge::for_each_by_case(dim, rb, re, o.begin(), o.end(),
                            [&](auto idx, auto x, auto y) ALWAYS_INLINE {ret -= (x + rsaimul) * std::log(orsaimul + y);},
                            [&](auto idx, auto x) ALWAYS_INLINE {ret -= (x + rsaimul) * lorsaimul;},
                            [&](auto idx, auto y) ALWAYS_INLINE {ret -= rsaimul * std::log(orsaimul + y);});
                        ret -= sharedz * rsaimul * lorsaimul;
                    } else {
                        while(rb != re) {
                            while(ind < rb->index())
                                ret -= rsaimul * std::log(o[ind++] + orsaimul);
                            ret -= (rb->value() + rsaimul) * std::log(o[ind]);
                        }
                        while(ind < r.size()) ret -= rsaimul * std::log(o[ind++] + orsaimul);
                    }
                } else {
                    throw TODOError("TODO: Special fast version for sparsity-preserving with non-uniform prior");
                }
            }
        } else {
            ret -= blaze::dot(row(i), blaze::neginf2zero(blaze::log(o)));
        }
        return ret;
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    FT mkl(const OT &o, size_t i, const OT2 &olog) const {
        if(IS_SPARSE && blaze::IsSparseVector_v<OT> && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        return blaze::dot(o, olog - logrow(i));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    FT mkl(const OT &o, size_t i) const {
        if(IS_SPARSE && blaze::IsSparseVector_v<OT> && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        return blaze::dot(o, blaze::neginf2zero(blaze::log(o)) - logrow(i));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    FT mkl(size_t i, const OT &, const OT2 &olog) const {
        if(IS_SPARSE && blaze::IsSparseVector_v<OT> && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        return blaze::dot(row(i), logrow(i) - olog);
    }
    template<typename...Args>
    FT pkl(Args &&...args) const { return mkl(std::forward<Args>(args)...);}
    template<typename...Args>
    FT psd(Args &&...args) const { return jsd(std::forward<Args>(args)...);}
    template<typename...Args>
    FT psm(Args &&...args) const { return jsm(std::forward<Args>(args)...);}
    FT bhattacharyya_sim(size_t i, size_t j) const {
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        return sqrdata_.rows() ? blaze::dot(sqrtrow(i), sqrtrow(j))
                               : blaze::sum(blaze::sqrt(row(i) * row(j)));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    FT bhattacharyya_sim(size_t i, const OT &o, const OT2 &osqrt) const {
        if(IS_SPARSE && prior_data_) throw std::runtime_error("Failed to calculate. TODO: complete special fast version of this supporting priors at no runtime cost.");
        return sqrdata_.rows() ? blaze::dot(sqrtrow(i), osqrt)
                               : blaze::sum(blaze::sqrt(row(i) * o));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    FT bhattacharyya_sim(size_t i, const OT &o) const {
        if(IS_SPARSE && prior_data_) throw std::runtime_error("Failed to calculate. TODO: complete special fast version of this supporting priors at no runtime cost.");
        return bhattacharyya_sim(i, o, blaze::sqrt(o));
    }
    template<typename...Args>
    FT bhattacharyya_distance(Args &&...args) const {
        if(IS_SPARSE && prior_data_) throw std::runtime_error("Failed to calculate. TODO: complete special fast version of this supporting priors at no runtime cost.");
        return -std::log(bhattacharyya_sim(std::forward<Args>(args)...));
    }
    template<typename...Args>
    FT bhattacharyya_metric(Args &&...args) const {
        if(IS_SPARSE && prior_data_) throw std::runtime_error("Failed to calculate. TODO: complete special fast version of this supporting priors at no runtime cost.");
        return std::sqrt(1 - bhattacharyya_sim(std::forward<Args>(args)...));
    }
    FT llr(size_t i, size_t j) const {
        if constexpr(IS_SPARSE) {
            if(prior_data_)
                return __llr_sparse_prior(i, j);
        }
            //blaze::dot(row(i), logrow(i)) * row_sums_[i]
            //+
            //blaze::dot(row(j), logrow(j)) * row_sums_[j]
            // X_j^Tlog(p_j)
            // X_k^Tlog(p_k)
            // (X_k + X_j)^Tlog(p_jk)
        const auto lhn = row_sums_[i], rhn = row_sums_[j];
        const auto lambda = lhn / (lhn + rhn), m1l = 1. - lambda;
        FT ret = lhn * __getjsc(i) + rhn * __getjsc(j)
            -
            blaze::dot(weighted_row(i) + weighted_row(j),
                neginf2zero(blaze::log(lambda * row(i) + m1l * row(j)))
            );
        assert(ret >= -1e-2 * (row_sums_[i] + row_sums_[j]) || !std::fprintf(stderr, "ret: %g\n", ret));
        return std::max(ret, FT(0.));
    }
    FT ollr(size_t i, size_t j) const {
        if(IS_SPARSE && prior_data_) {
            std::fprintf(stderr, "note: ollr with prior is slightly incorrect due to the sparsity-destroying nature of the prior.\n");
        }
        FT ret = __getjsc(i) * row_sums_[i] + __getjsc(j) * row_sums_[j]
            - blaze::dot(weighted_row(i) + weighted_row(j), neginf2zero(blaze::log((row(i) + row(j)) * .5)));
        return std::max(ret, FT(0.));
    }
    FT uwllr(size_t i, size_t j) const {
        if(IS_SPARSE && prior_data_) {
            return __uwllr_sparse_prior(i, j);
        }
        const auto lhn = row_sums_[i], rhn = row_sums_[j];
        const auto lambda = lhn / (lhn + rhn), m1l = 1. - lambda;
        const auto rowprod = evaluate(lambda * row(i) + m1l * row(j));
        return std::max(lambda * __getjsc(i) + m1l * __getjsc(j) - blaze::dot(rowprod, neginf2zero(blaze::log(rowprod))), 0.);
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    FT llr(size_t, const OT &) const {
        throw TODOError("llr is not implemented for this.");
        return 0.;
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    FT llr(size_t, const OT &, const OT2 &) const {
        throw TODOError("llr is not implemented for this.");
        return 0.;
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    FT uwllr(size_t i, const OT &o) const {
        if constexpr(IS_SPARSE) {
            if(prior_data_) {
                
            }
        }
        return 0.;
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    FT uwllr(size_t, const OT &, const OT2 &) const {
        throw TODOError("llr is not implemented for this.");
        return 0.;
    }
    template<typename...Args>
    FT jsm(Args &&...args) const {
        return std::sqrt(jsd(std::forward<Args>(args)...));
    }
    auto get_measure() const {return measure_;}
private:
    static constexpr FT PI_INV = 1. / 3.14159265358979323846264338327950288;

    template<typename Container=blaze::DynamicVector<FT, blaze::rowVector>>
    void prep(Prior prior, const Container *c=nullptr) {
        switch(prior) {
            case NONE:
            break;
            case DIRICHLET:
                if constexpr(!IsSparseMatrix_v<MatrixType>) {
                    data_ += static_cast<FT>(1);
                } else {
                    prior_data_.reset(new VecT({FT(1)}));
                }
                break;
            case GAMMA_BETA:
                if(c == nullptr) throw std::invalid_argument("Can't do gamma_beta with null pointer");
                if constexpr(IsSparseMatrix_v<MatrixType>) {
                    prior_data_.reset(new VecT({(*c)[0]}));
                } else if constexpr(IsDenseMatrix_v<MatrixType>) {
                    data_ += (*c)[0];
                }
            break;
            case FEATURE_SPECIFIC_PRIOR:
                if(c == nullptr) throw std::invalid_argument("Can't do feature-specific with null pointer");
                if constexpr(IsDenseMatrix_v<MatrixType>) {
                    data_ += blaze::expand(*c, data_.rows());
                } else if constexpr(IsSparseMatrix_v<MatrixType>) {
                    assert(c->size() == data_.columns());
                    prior_data_.reset(new VecT(data_.columns()));
                    *prior_data_ = *c;
                }
            break;
        }
        if(prior_data_) {
            if(prior_data_->size() == 1) prior_sum_ = data_.columns() * prior_data_->operator[](0);
            else                         prior_sum_ = std::accumulate(prior_data_->begin(), prior_data_->end(), 0.);
        }
        row_sums_.resize(data_.rows());
        {
            for(size_t i = 0; i < data_.rows(); ++i) {
                auto r(row(i));
                FT countsum = blaze::sum(r);
                if constexpr(blaze::IsDenseMatrix_v<MatrixType>) {
                    if(prior == NONE) {
                        r += 1e-50;
                        if(dist::detail::expects_nonnegative(measure_) && blaze::min(r) < 0.)
                            throw std::invalid_argument(std::string("Measure ") + dist::detail::prob2str(measure_) + " expects nonnegative data");
                    }
                }
                FT div = countsum;
                if constexpr(blaze::IsSparseMatrix_v<MatrixType>) {
                    if(prior_data_)
                        div += prior_sum_;
                }
                r /= div;
                row_sums_[i] = countsum;
            }
        }

        if(dist::detail::needs_logs(measure_)) {
            if constexpr(IS_CSC_VIEW) {
                logdata_ = CacheMatrixType(data_.rows(), data_.columns());
                logdata_.reserve(data_.nnz());
                for(size_t i = 0; i < data_.rows(); ++i) {
                    for(const auto &pair: data_.column(i)) {
                        logdata_.set(i, pair.index(), std::log(pair.value()));
                    }
                }
            } else {
                logdata_ = CacheMatrixType(neginf2zero(log(data_)));
            }
        }
        if(dist::detail::needs_sqrt(measure_)) {
            if constexpr(IS_CSC_VIEW) {
                sqrdata_ = CacheMatrixType(data_.rows(), data_.columns());
                sqrdata_.reserve(data_.nnz());
                for(size_t i = 0; i < data_.rows(); ++i) {
                    for(const auto &pair: data_.column(i)) {
                        sqrdata_.set(i, pair.index(), std::sqrt(pair.value()));
                    }
                }
            } else {
                sqrdata_ = CacheMatrixType(blaze::sqrt(data_));
            }
        }
        if(dist::detail::needs_l2_cache(measure_)) {
            l2norm_cache_.reset(new VecT(data_.rows()));
            OMP_PFOR
            for(size_t i = 0; i < data_.rows(); ++i) {
                if(prior_data_ && IS_SPARSE) {
                    auto &pd = *prior_data_;
                    double s = 0.;
                    auto r = weighted_row(i);
                    auto getv = [&](size_t x) {return pd.size() == 1 ? pd[0]: pd[x];};
                    for(size_t i = 0; i < data_.columns(); ++i) {
                        s += std::pow(r[i] + getv(i), 2);
                    }
                    l2norm_cache_->operator[](i)  = 1. / std::sqrt(s);
                } else {
                    l2norm_cache_->operator[](i)  = 1. / blaze::l2Norm(weighted_row(i));
                }
            }
        }
        if(dist::detail::needs_probability_l2_cache(measure_)) {
            pl2norm_cache_.reset(new VecT(data_.rows()));
            OMP_PFOR
            for(size_t i = 0; i < data_.rows(); ++i) {
                if(prior_data_ && IS_SPARSE) {
                    auto &pd = *prior_data_;
                    auto rsi = 1. / (prior_sum_ + row_sums_[i]);
                    blz::DV<double, blz::rowVector> tmp = weighted_row(i);
                    if(pd.size() == 1) tmp += pd[0];
                    else               tmp += pd;
                    pl2norm_cache_->operator[](i)  = 1. / blaze::l2Norm(tmp * rsi);
                } else {
                    pl2norm_cache_->operator[](i)  = 1. / blaze::l2Norm(weighted_row(i));
                }
            }
        }
        if(logdata_.rows()) {
            jsd_cache_.resize(data_.rows());
            auto &jc = jsd_cache_;
            if constexpr(IS_SPARSE) {
                if(prior_data_) {
                    // Handle sparse priors
                    MINOCORE_VALIDATE(prior_data_->size() == 1 || prior_data_->size() == data_.columns());
                    auto &pd = *prior_data_;
                    const bool single_value = pd.size() == 1;
                    for(size_t i = 0; i < data_.rows(); ++i) {
                        const auto rs = row_sums_[i] + prior_sum_;
                        auto r = row(i);
                        double contrib = 0.;
                        auto upcontrib = [&](auto x) {contrib += x * std::log(x);};
                        if(single_value) {
                            FT invp = pd[0] / rs;
                            size_t number_zero = r.size() - nonZeros(r);
                            contrib += number_zero * (invp * std::log(invp)); // Empty
                            for(auto &pair: r) upcontrib(pair.value() + invp);       // Non-empty
                        } else {
                            size_t i = 0;
                            auto it = r.begin();
                            auto contribute_range = [&](size_t end) {
                                while(i < end) upcontrib(pd[i++] / rs);
                            };
                            while(it != r.end() && i < r.size()) {
                                auto idx = it->index();
                                contribute_range(idx);
                                upcontrib(it->value() + pd[idx] / rs);
                                if(++it == r.end())
                                    contribute_range(r.size());
                            }
                        }
                        jc[i] = contrib;
                    }
                }
            }
            if(!(IS_SPARSE && prior_data_))
                for(size_t i = 0; i < jc.size(); ++i)
                    jc[i] = dot(row(i), logrow(i));
        }
    }
    INLINE FT __getjsc(size_t index) const {
        assert(index < jsd_cache_.size());
        return jsd_cache_[index];
    }
    INLINE FT __getlsc(size_t index) const {
        return __getjsc(index) * row_sums_->operator[](index);
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>> >
    FT __llr_sparse_prior(size_t i, const OT &o) const {
        assert(IS_SPARSE);
        auto lhr(row(i));
        const auto lhn = row_sums_[i] + prior_sum_, rhn = blz::sum(o);
        const auto lhrsi = 1. / lhn, rhrsi = 1. / rhn;
        const auto lambda = lhn / (lhn + rhn), m1l = 1. - lambda;
        const auto &pd(*prior_data_);
        FT ret = lhn * __getjsc(i);
        if(pd.size() == 1) {
            const auto pv = pd[0], pv2 = pv * 2;
            const auto lhrsimul = lhrsi * pv;
            const auto rhrsimul = rhrsi * pv;
            size_t shared_zero = merge::for_each_by_case(lhr.size(), lhr.begin(), lhr.end(), o.begin(), o.end(),
                [&](auto, auto x, auto y) {
                    const auto xcontr = lhn * x + pv;
                    const auto ycontr = rhn * y + pv;
                    const auto nlogv = -std::log(lambda * (x + lhrsimul) + m1l * (y + rhrsimul));
                    ret += xcontr * nlogv + ycontr * (std::log(y + rhrsimul) + nlogv);
#if 0
                    ret -= (xcontr + ycontr) * logv;
                    ret += ycontr * std::log(y + rhrsimul);
#endif
                },
                [&](auto, auto x) {ret -= (lhn * x + pv2) * std::log(lambda * (x + lhrsimul) + m1l * rhrsimul);},
                [&](auto, auto y) {
                    ret -= (rhn * y + pv2) * std::log(lambda * lhrsimul + m1l * (y + rhrsimul));
                    ret += (lhn * y + pv) * std::log(y + rhrsimul);
                });
            auto prod1 = -pv2 * std::log(lambda * lhrsimul + m1l * rhrsimul); // from normal derivation
            auto prod2 = pv * std::log(rhrsi); // from o's self-contributions
            ret += shared_zero * (prod1 + prod2);
        } else {
            throw NotImplementedError("llr sparse prior, external data");
        }
        return std::max(ret, FT(0.));
    }
    FT __llr_sparse_prior(size_t i, size_t j) const {
        assert(IS_SPARSE);
        auto lhr(row(i)), rhr(row(j));
        const auto lhn = row_sums_[i] + prior_sum_, rhn = row_sums_[j] + prior_sum_;
        const auto lhrsi = 1. / lhn, rhrsi = 1. / rhn;
        const auto lambda = lhn / (lhn + rhn), m1l = 1. - lambda;
        const auto &pd(*prior_data_);
        FT ret = lhn * __getjsc(i) + rhn * __getjsc(j);
        if(pd.size() == 1) {
            const auto pv = pd[0], pv2 = pv * 2;
            const auto lhrsimul = lhrsi * pv;
            const auto rhrsimul = rhrsi * pv;
            size_t shared_zero = merge::for_each_by_case(lhr.size(), lhr.begin(), lhr.end(), rhr.begin(), rhr.end(),
                [&](auto, auto x, auto y) {ret -= (lhn * x + rhn * y + pv2) * std::log(lambda * (x + lhrsimul) + m1l * (y + rhrsimul));},
                [&](auto, auto x) {ret -= (lhn * x + pv2) * std::log(lambda * (x + lhrsimul) + m1l * rhrsimul);},
                [&](auto, auto y) {ret -= (rhn * y + pv2) * std::log(lambda * lhrsimul + m1l * (y + rhrsimul));});
            ret -= shared_zero * pv2 * std::log(lambda * lhrsimul + m1l * rhrsimul);
        } else {
            merge::for_each_by_case(lhr.size(), lhr.begin(), lhr.end(), rhr.begin(), rhr.end(),
                [&](auto idx, auto x, auto y) {ret -= (lhn * x + rhn * y + pd[idx] * 2.) * std::log(lambda * (x + pd[idx] * lhrsi) + m1l * (y + pd[idx] * rhrsi));},
                [&](auto idx, auto x) {ret -= (lhn * x + pd[idx] * 2.) * std::log(lambda * (x + pd[idx] * lhrsi) + m1l * pd[idx] * rhrsi);},
                [&](auto idx, auto y) {ret -= (rhn * y + pd[idx] * 2.) * std::log(lambda * pd[idx] * lhrsi + m1l * (y + pd[idx] * rhrsi));},
                [&](auto idx) {ret -= (pd[idx] * 2.) * std::log(lambda * pd[idx] * lhrsi + m1l * (pd[idx] * rhrsi));});
        }
        return std::max(ret, FT(0.));
    }
    FT __uwllr_sparse_prior(size_t i, size_t j) const {
        assert(IS_SPARSE);
        auto lhr(row(i)), rhr(row(j));
        const auto lhn = row_sums_[i] + prior_sum_, rhn = row_sums_[j] + prior_sum_;
        const auto bothn = lhn + rhn;
        const auto lhrsi = 1. / lhn, rhrsi = 1. / rhn;
        const auto lambda = lhn / (lhn + rhn), m1l = 1. - lambda;
        const auto &pd(*prior_data_);
        FT ret = lambda * __getjsc(i) + m1l * __getjsc(j);
        if(pd.size() == 1) {
            const auto pv = pd[0];
            const auto lhrsimul = lhrsi * pv;
            const auto rhrsimul = rhrsi * pv;
            const auto bothsimul = lambda * lhrsimul + m1l * rhrsimul;
            size_t shared_zero = merge::for_each_by_case(lhr.size(), lhr.begin(), lhr.end(), rhr.begin(), rhr.end(),
                [&](auto, auto x, auto y) {ret -= (x + y + bothsimul) * std::log(lambda * (x + lhrsimul) + m1l * (y + rhrsimul));},
                [&](auto, auto x) {ret -= (x + bothsimul) * std::log(lambda * (x + lhrsimul) + m1l * rhrsimul);},
                [&](auto, auto y) {ret -= (y + bothsimul) * std::log(lambda * lhrsimul + m1l * (y + rhrsimul));});
            ret -= shared_zero * bothsimul * std::log(bothsimul);
        } else {
            const auto bothsi2 = 2. / bothn; // Because the prior is added two both left and right hand side
            merge::for_each_by_case(lhr.size(), lhr.begin(), lhr.end(), rhr.begin(), rhr.end(),
                [&](auto idx, auto x, auto y) {ret -= (x + y + pd[idx] * bothsi2) * std::log(lambda * (x + pd[idx] * lhrsi) + m1l * (y + pd[idx] * rhrsi));},
                [&](auto idx, auto x) {ret -= (x + pd[idx] * bothsi2) * std::log(lambda * (x + pd[idx] * lhrsi) + m1l * pd[idx] * rhrsi);},
                [&](auto idx, auto y) {ret -= (y + pd[idx] * bothsi2) * std::log(lambda * pd[idx] * lhrsi + m1l * (y + pd[idx] * rhrsi));},
                [&](auto idx) {ret -= (pd[idx] * bothsi2) * std::log(pd[idx] * (lambda * lhrsi + m1l * rhrsi));});
        }
        return std::max(ret, FT(0.));
    }
}; // DissimilarityApplicator

template<typename T>
struct is_dissimilarity_applicator: std::false_type {};
template<typename MT>
struct is_dissimilarity_applicator<DissimilarityApplicator<MT>>: std::true_type {};
template<typename T>
static constexpr bool is_dissimilarity_applicator_v = is_dissimilarity_applicator<T>::value;

template<typename MT1, typename MT2>
struct PairDissimilarityApplicator {
    DissimilarityApplicator<MT1> &pda_;
    DissimilarityApplicator<MT2> &pdb_;
    PairDissimilarityApplicator(DissimilarityApplicator<MT1> &lhs, DissimilarityApplicator<MT2> &rhs): pda_(lhs), pdb_(rhs) {
        if(lhs.measure_ != rhs.measure_) throw std::runtime_error("measures must be the same (for preprocessing reasons).");
    }
    decltype(auto) operator()(size_t i, size_t j) const {
        return pda_(i, pdb_.row(j));
    }
};

template<typename MatrixType>
class MultinomialJSDApplicator: public DissimilarityApplicator<MatrixType> {
    using super = DissimilarityApplicator<MatrixType>;
    template<typename PriorContainer=blaze::DynamicVector<typename super::FT, blaze::rowVector>>
    MultinomialJSDApplicator(MatrixType &ref,
                             Prior prior=NONE,
                             const PriorContainer *c=nullptr):
        DissimilarityApplicator<MatrixType>(ref, JSD, prior, c) {}
};
template<typename MatrixType>
class MultinomialLLRApplicator: public DissimilarityApplicator<MatrixType> {
    using super = DissimilarityApplicator<MatrixType>;
    template<typename PriorContainer=blaze::DynamicVector<typename super::FT, blaze::rowVector>>
    MultinomialLLRApplicator(MatrixType &ref,
                             Prior prior=NONE,
                             const PriorContainer *c=nullptr):
        DissimilarityApplicator<MatrixType>(ref, LLR, prior, c) {}
};

template<typename MatrixType, typename PriorContainer=blaze::DynamicVector<typename MatrixType::ElementType, blaze::rowVector>>
auto make_probdiv_applicator(MatrixType &data, DissimilarityMeasure type=JSM, Prior prior=NONE, const PriorContainer *pc=nullptr) {
    return DissimilarityApplicator<MatrixType>(data, type, prior, pc);
}
template<typename MatrixType, typename PriorContainer=blaze::DynamicVector<typename MatrixType::ElementType, blaze::rowVector>>
auto make_jsm_applicator(MatrixType &data, Prior prior=NONE, const PriorContainer *pc=nullptr) {
    return make_probdiv_applicator(data, JSM, prior, pc);
}


template<typename MatrixType>
auto make_kmc2(const DissimilarityApplicator<MatrixType> &app, unsigned k, size_t m=2000, uint64_t seed=13) {
    wy::WyRand<uint64_t> gen(seed);
    return coresets::kmc2(app, gen, app.size(), k, m);
}

template<typename MatrixType, typename WFT=blaze::ElementType_t<MatrixType>>
auto make_kmeanspp(const DissimilarityApplicator<MatrixType> &app, unsigned k, uint64_t seed=13, const WFT *weights=nullptr) {
    wy::WyRand<uint64_t> gen(seed);
    return coresets::kmeanspp(app, gen, app.size(), k, weights);
}

template<typename MatrixType, typename WFT=blaze::ElementType_t<MatrixType>>
auto make_kcenter(const DissimilarityApplicator<MatrixType> &app, unsigned k, uint64_t seed=13, const WFT *weights=nullptr) {
    wy::WyRand<uint64_t> gen(seed);
    return coresets::kmeanspp(app, gen, app.size(), k, weights);
}

template<typename MatrixType, typename WFT=typename MatrixType::ElementType, typename IT=uint32_t>
auto make_d2_coreset_sampler(const DissimilarityApplicator<MatrixType> &app, unsigned k, uint64_t seed=13, const WFT *weights=nullptr, coresets::SensitivityMethod sens=cs::LBK) {
    auto [centers, asn, costs] = make_kmeanspp(app, k, seed);
    coresets::CoresetSampler<typename MatrixType::ElementType, IT> cs;
    cs.make_sampler(app.size(), centers.size(), costs.data(), asn.data(), weights,
                    seed + 1, sens);
    return cs;
}

} // namespace cmp

namespace jsd = cmp; // until code change is complete.

using jsd::DissimilarityApplicator;
using jsd::make_d2_coreset_sampler;
using jsd::make_kmc2;
using jsd::make_kmeanspp;
using jsd::make_jsm_applicator;
using jsd::make_probdiv_applicator;



} // minocore

#endif
