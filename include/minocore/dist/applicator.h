#ifndef FGC_JSD_H__
#define FGC_JSD_H__
#include "minocore/util/exception.h"
#include "minocore/coreset.h"
#include "minocore/dist/distance.h"
#include "distmat/distmat.h"
#include "minocore/optim/kmeans.h"
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/polygamma.hpp>
#include <set>


namespace minocore {

namespace jsd {

using namespace blz;
using namespace blz::distance;

template<typename MatrixType>
class DissimilarityApplicator {
    //using opposite_type = typename base_type::OppositeType;
    MatrixType &data_;
    using VecT = blaze::DynamicVector<typename MatrixType::ElementType, IsRowMajorMatrix_v<MatrixType> ? blaze::rowVector: blaze::columnVector>;
    using matrix_type = MatrixType;
    VecT row_sums_;
    std::unique_ptr<MatrixType> logdata_;
    std::unique_ptr<MatrixType> sqrdata_;
    std::unique_ptr<VecT> jsd_cache_;
    std::unique_ptr<VecT> prior_data_;
    std::unique_ptr<VecT> l2norm_cache_;
    std::unique_ptr<VecT> pl2norm_cache_;
    typename MatrixType::ElementType lambda_ = 0.5;
    static constexpr bool IS_SPARSE      = IsSparseMatrix_v<MatrixType>;
    static constexpr bool IS_DENSE_BLAZE = IsDenseMatrix_v<MatrixType>;
public:
    using FT = typename MatrixType::ElementType;
    using MT = MatrixType;
    using This = DissimilarityApplicator<MatrixType>;
    using ConstThis = const DissimilarityApplicator<MatrixType>;

    const DissimilarityMeasure measure_;
    const MatrixType &data() const {return data_;}
    const VecT &row_sums() const {return row_sums_;}
    size_t size() const {return data_.rows();}
    template<typename PriorContainer=blaze::DynamicVector<FT, blaze::rowVector>>
    DissimilarityApplicator(MatrixType &ref,
                      DissimilarityMeasure measure=JSM,
                      Prior prior=NONE,
                      const PriorContainer *c=nullptr):
        data_(ref), logdata_(nullptr), measure_(measure)
    {
        prep(prior, c);
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
            //std::fprintf(stderr, "Symmetric measure %s/%s\n", detail::prob2str(measure), detail::prob2desc(measure));
            if(symmetrize) {
                fill_symmetric_upper_triangular(m);
            }
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
            default: throw std::invalid_argument(std::string("unknown dissimilarity measure: ") + std::to_string(int(measure)) + blz::detail::prob2str(measure));
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
    auto cosine_similarity(size_t i, size_t j) const {
        return blaze::dot(weighted_row(i), weighted_row(j)) * l2norm_cache_->operator[](i) * l2norm_cache_->operator[](j);
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    auto cosine_similarity(size_t j, const OT &o) const {
        return blaze::dot(o, weighted_row(j)) / blz::l2Norm(o) * l2norm_cache_->operator[](j);
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    auto cosine_similarity(const OT &o, size_t j) const {
        return blaze::dot(o, weighted_row(j)) / blz::l2Norm(o) * l2norm_cache_->operator[](j);
    }
    auto pcosine_similarity(size_t i, size_t j) const {
        return blaze::dot(row(i), row(j)) * pl2norm_cache_->operator[](i) * pl2norm_cache_->operator[](j);
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    auto pcosine_similarity(size_t j, const OT &o) const {
        return blaze::dot(o, row(j)) / blz::l2Norm(o) * pl2norm_cache_->operator[](j);
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    auto pcosine_similarity(const OT &o, size_t j) const {
        return blaze::dot(o, row(j)) / blz::l2Norm(o) * pl2norm_cache_->operator[](j);
    }

    static constexpr FT PI_INV = 1. / 3.14159265358979323846264338327950288;

    template<typename...Args>
    auto cosine_distance(Args &&...args) const {
        return std::acos(cosine_similarity(std::forward<Args>(args)...)) * PI_INV;
    }
    template<typename...Args>
    auto pcosine_distance(Args &&...args) const {
        return std::acos(pcosine_similarity(std::forward<Args>(args)...)) * PI_INV;
    }
    auto dotproduct_distance(size_t i, size_t j) const {
        return blaze::dot(weighted_row(i), weighted_row(j)) * l2norm_cache_->operator[](i) * l2norm_cache_->operator[](j);
    }
    auto pdotproduct_distance(size_t i, size_t j) const {
        return blaze::dot(row(i), row(j)) * pl2norm_cache_->operator[](i) * pl2norm_cache_->operator[](j);
    }


    // Accessors
    decltype(auto) weighted_row(size_t ind) const {
        return blaze::row(data_, ind BLAZE_CHECK_DEBUG) * row_sums_[ind];
    }
    auto row(size_t ind) const {return blaze::row(data_, ind BLAZE_CHECK_DEBUG);}
    auto logrow(size_t ind) const {return blaze::row(*logdata_, ind BLAZE_CHECK_DEBUG);}
    auto sqrtrow(size_t ind) const {return blaze::row(*sqrdata_, ind BLAZE_CHECK_DEBUG);}


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
            ret = p_wasserstein(row(i), o);
        } else if constexpr(constexpr_measure == WEMD) {
            ret = p_wasserstein(weighted_row(i), o);
        } else if constexpr(constexpr_measure == REVERSE_POISSON) {
            ret = cp ? pkl(i, o, *cp): pkl(i, o);
        } else if constexpr(constexpr_measure == POISSON) {
            ret = cp ? pkl(o, i, *cp): pkl(o, i);
        } else if constexpr(constexpr_measure == HELLINGER) {
            ret = cp ? blz::sqrNorm(sqrtrow(i) - *cp)
                     : blz::sqrNorm(sqrtrow(i) - blz::sqrt(o));
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
    INLINE FT call(size_t i, OT &o, CacheT *cp=static_cast<CacheT *>(nullptr)) const {
        FT ret;
        if constexpr(constexpr_measure == TOTAL_VARIATION_DISTANCE) {
            ret = discrete_total_variation_distance(row(i), o);
        } else if constexpr(constexpr_measure == L1) {
            ret = l1Norm(weighted_row(i) - o);
        } else if constexpr(constexpr_measure == L2) {
            ret = l2Norm(weighted_row(i) - o);
        } else if constexpr(constexpr_measure == SQRL2) {
            ret = blaze::sqrNorm(weighted_row(i) - o);
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
            ret = p_wasserstein(row(i), o);
        } else if constexpr(constexpr_measure == WEMD) {
            ret = p_wasserstein(weighted_row(i), o);
        } else if constexpr(constexpr_measure == REVERSE_POISSON) {
            ret = cp ? pkl(o, i, *cp): pkl(o, i);
        } else if constexpr(constexpr_measure == POISSON) {
            ret = cp ? pkl(i, o, *cp): pkl(i, o);
        } else if constexpr(constexpr_measure == HELLINGER) {
            if(cp) {
                ret = blz::sqrNorm(sqrtrow(i) - *cp);
            } else {
                ret = blz::sqrNorm(sqrtrow(i) - blz::sqrt(o));
            }
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
        } else if constexpr(constexpr_measure == SQRL2) {
            ret = blaze::sqrNorm(weighted_row(i) - weighted_row(j));
        } else if constexpr(constexpr_measure == JSD) {
            ret = jsd(i, j);
        } else if constexpr(constexpr_measure == JSM) {
            ret = jsm(i, j);
        } else if constexpr(constexpr_measure == REVERSE_MKL) {
            ret = mkl(j, i);
        } else if constexpr(constexpr_measure == MKL) {
            ret = mkl(i, j);
        } else if constexpr(constexpr_measure == EMD) {
            ret = p_wasserstein(row(i), row(j));
        } else if constexpr(constexpr_measure == WEMD) {
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
    INLINE FT operator()(const OT &o, size_t i, const CacheT *cache=static_cast<CacheT *>(nullptr), DissimilarityMeasure measure=static_cast<DissimilarityMeasure>(-1)) const noexcept {
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
            case TOTAL_VARIATION_DISTANCE: ret = call<TOTAL_VARIATION_DISTANCE>(o, i); break;
            case L1: ret = call<L1>(o, i); break;
            case L2: ret = call<L2>(o, i); break;
            case SQRL2: ret = call<SQRL2>(o, i); break;
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
    }
    template<typename OT, typename CacheT=OT, typename=std::enable_if_t<!std::is_integral_v<OT> > >
    INLINE FT operator()(size_t i, const OT &o, const CacheT *cache=static_cast<CacheT *>(nullptr), DissimilarityMeasure measure=static_cast<DissimilarityMeasure>(-1)) const noexcept {
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
    template<typename MatType>
    void operator()(MatType &mat, bool symmetrize=false) {
        set_distance_matrix(mat, symmetrize);
    }
    template<typename MatType>
    auto operator()() {
        return make_distance_matrix(measure_);
    }

    auto itakura_saito(size_t i, size_t j) const {
        FT ret;
        if constexpr(IS_SPARSE) {
            if(!prior_data_) {
                char buf[128];
                std::sprintf(buf, "warning: Itakura-Saito cannot be computed to sparse vectors/matrices at %zu/%zu\n", i, j);
                throw std::runtime_error(buf);
            }
            ret = -std::numeric_limits<FT>::max();
            throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        } else {
            auto div = row(i) / row(j);
            ret = blaze::sum(div - blaze::log(div)) - row(i).size();
        }
        return ret;
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>> >
    auto itakura_saito(size_t i, const OT &o) const {
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
            auto div = row(i) / o;
            ret = blaze::sum(div - blaze::log(div)) - row(i).size();
        }
        return ret;
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>> >
    auto itakura_saito(const OT &o, size_t i) const {
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

    auto hellinger(size_t i, size_t j) const {
        return sqrdata_ ? blaze::sqrNorm(sqrtrow(i) - sqrtrow(j))
                        : blaze::sqrNorm(blaze::sqrt(row(i)) - blaze::sqrt(row(j)));
    }
    FT jsd(size_t i, size_t j) const {
        if(!IsSparseMatrix_v<MatrixType> || !prior_data_) {
            assert(i < data_.rows());
            assert(j < data_.rows());
            FT ret;
            auto ri = row(i), rj = row(j);
            //constexpr FT logp5 = -0.693147180559945; // std::log(0.5)
            auto s = ri + rj;
            ret = jsd_cache_->operator[](i) + jsd_cache_->operator[](j) - blaze::dot(s, blaze::neginf2zero(blaze::log(s * 0.5)));
#ifndef NDEBUG
            static constexpr typename MatrixType::ElementType threshold
                = std::is_same_v<typename MatrixType::ElementType, double>
                                    ? 0.: -1e-5;
            assert(ret >= threshold || !std::fprintf(stderr, "ret: %g (numerical stability issues)\n", ret));
#endif
            return std::max(ret, static_cast<FT>(0.));
        } else {
            throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
            return FT(0);
        }
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    auto jsd(size_t i, const OT &o, const OT2 &olog) const {
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        auto mnlog = evaluate(log(0.5 * (row(i) + o)));
        return (blaze::dot(row(i), logrow(i) - mnlog) + blaze::dot(o, olog - mnlog));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    auto jsd(size_t i, const OT &o) const {
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        auto olog = evaluate(blaze::neginf2zero(blaze::log(o)));
        return jsd(i, o, olog);
    }
    auto mkl(size_t i, size_t j) const {
        // Multinomial KL
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        return get_jsdcache(i) - blaze::dot(row(i), logrow(j));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    auto mkl(size_t i, const OT &o) const {
        // Multinomial KL
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        return get_jsdcache(i) - blaze::dot(row(i), blaze::neginf2zero(blaze::log(o)));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    auto mkl(const OT &o, size_t i, const OT2 &olog) const {
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        // Multinomial KL
        return blaze::dot(o, olog - logrow(i));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    auto mkl(const OT &o, size_t i) const {
        // Multinomial KL
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        return blaze::dot(o, blaze::neginf2zero(blaze::log(o)) - logrow(i));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    auto mkl(size_t i, const OT &, const OT2 &olog) const {
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        // Multinomial KL
        return blaze::dot(row(i), logrow(i) - olog);
    }
    auto pkl(size_t i, size_t j) const {
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        // Poission KL
        return get_jsdcache(i) - blaze::dot(row(i), logrow(j)) + blaze::sum(row(j) - row(i));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    auto pkl(size_t i, const OT &o, const OT2 &olog) const {
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        // Poission KL
        return get_jsdcache(i) - blaze::dot(row(i), olog) + blaze::sum(row(i) - o);
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    auto pkl(size_t i, const OT &o) const {
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        return pkl(i, o, neginf2zero(blaze::log(o)));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    auto pkl(const OT &o, size_t i) const {
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        throw TODOError("Not done\n");
        return pkl(i, o, neginf2zero(blaze::log(o)));
    }
    template<typename OT, typename OT2, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    auto pkl(const OT &o, size_t i, const OT2 &cache) const {
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        throw TODOError("Not done\n");
        return pkl(i, o, neginf2zero(blaze::log(o)));
    }
    auto psd(size_t i, size_t j) const {
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        // Poission JSD
        auto mnlog = evaluate(log(.5 * (row(i) + row(j))));
        return (blaze::dot(row(i), logrow(i) - mnlog) + blaze::dot(row(j), logrow(j) - mnlog));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    auto psd(size_t i, const OT &o, const OT2 &olog) const {
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        // Poission JSD
        auto mnlog = evaluate(log(.5 * (row(i) + o)));
        return (blaze::dot(row(i), logrow(i) - mnlog) + blaze::dot(o, olog - mnlog));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    auto psd(size_t i, const OT &o) const {
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        return psd(i, o, neginf2zero(blaze::log(o)));
    }
    auto bhattacharyya_sim(size_t i, size_t j) const {
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        return sqrdata_ ? blaze::dot(sqrtrow(i), sqrtrow(j))
                        : blaze::sum(blaze::sqrt(row(i) * row(j)));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    auto bhattacharyya_sim(size_t i, const OT &o, const OT2 &osqrt) const {
        if(IS_SPARSE && prior_data_) throw std::runtime_error("Failed to calculate. TODO: complete special fast version of this supporting priors at no runtime cost.");
        return sqrdata_ ? blaze::dot(sqrtrow(i), osqrt)
                        : blaze::sum(blaze::sqrt(row(i) * o));
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    auto bhattacharyya_sim(size_t i, const OT &o) const {
        if(IS_SPARSE && prior_data_) throw std::runtime_error("Failed to calculate. TODO: complete special fast version of this supporting priors at no runtime cost.");
        return bhattacharyya_sim(i, o, blaze::sqrt(o));
    }
    template<typename...Args>
    auto bhattacharyya_distance(Args &&...args) const {
        if(IS_SPARSE && prior_data_) throw std::runtime_error("Failed to calculate. TODO: complete special fast version of this supporting priors at no runtime cost.");
        return -std::log(bhattacharyya_sim(std::forward<Args>(args)...));
    }
    template<typename...Args>
    auto bhattacharyya_metric(Args &&...args) const {
        if(IS_SPARSE && prior_data_) throw std::runtime_error("Failed to calculate. TODO: complete special fast version of this supporting priors at no runtime cost.");
        return std::sqrt(1 - bhattacharyya_sim(std::forward<Args>(args)...));
    }
    template<typename...Args>
    auto psm(Args &&...args) const {return std::sqrt(std::forward<Args>(args)...);}
    auto llr(size_t i, size_t j) const {
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
            //blaze::dot(row(i), logrow(i)) * row_sums_[i]
            //+
            //blaze::dot(row(j), logrow(j)) * row_sums_[j]
            // X_j^Tlog(p_j)
            // X_k^Tlog(p_k)
            // (X_k + X_j)^Tlog(p_jk)
        const auto lhn = row_sums_[i], rhn = row_sums_[j];
        const auto lambda = lhn / (lhn + rhn), m1l = 1. - lambda;
        auto ret = lhn * get_jsdcache(i) + rhn * get_jsdcache(j)
            -
            blaze::dot(weighted_row(i) + weighted_row(j),
                neginf2zero(blaze::log(lambda * row(i) + m1l * row(j)))
            );
        assert(ret >= -1e-2 * (row_sums_[i] + row_sums_[j]) || !std::fprintf(stderr, "ret: %g\n", ret));
        return std::max(ret, 0.);
    }
    auto ollr(size_t i, size_t j) const {
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        auto ret = get_jsdcache(i) * row_sums_[i] + get_jsdcache(j) * row_sums_[j]
            - blaze::dot(weighted_row(i) + weighted_row(j), neginf2zero(blaze::log((row(i) + row(j)) * .5)));
        return std::max(ret, 0.);
    }
    auto uwllr(size_t i, size_t j) const {
        if(IS_SPARSE && prior_data_) throw TODOError("TODO: complete special fast version of this supporting priors at no runtime cost.");
        const auto lhn = row_sums_[i], rhn = row_sums_[j];
        const auto lambda = lhn / (lhn + rhn), m1l = 1. - lambda;
        return
          std::max(
            lambda * get_jsdcache(i) +
                  m1l * get_jsdcache(j) -
               blaze::dot(lambda * row(i) + m1l * row(j),
                        neginf2zero(blaze::log(
                            lambda * row(i) + m1l * row(j)))),
          0.);
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    auto llr(size_t, const OT &) const {
        throw TODOError("llr is not implemented for this.");
        return 0.;
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    auto llr(size_t, const OT &, const OT2 &) const {
        throw TODOError("llr is not implemented for this.");
        return 0.;
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>>
    auto uwllr(size_t, const OT &) const {
        throw TODOError("llr is not implemented for this.");
        return 0.;
    }
    template<typename OT, typename=std::enable_if_t<!std::is_integral_v<OT>>, typename OT2>
    auto uwllr(size_t, const OT &, const OT2 &) const {
        throw TODOError("llr is not implemented for this.");
        return 0.;
    }
    template<typename...Args>
    auto jsm(Args &&...args) const {
        return std::sqrt(jsd(std::forward<Args>(args)...));
    }
    void set_lambda(FT param) {
        if(param < 0. || param > 1.)
            throw std::invalid_argument(std::string("Param for lambda ") + std::to_string(param) + " is out of range.");
        lambda_ = param;
    }
private:
    template<typename Container=blaze::DynamicVector<FT, blaze::rowVector>>
    void prep(Prior prior, const Container *c=nullptr) {
        switch(prior) {
            case NONE:
            break;
            case DIRICHLET:
                if constexpr(!IsSparseMatrix_v<MatrixType>) {
                    data_ += static_cast<FT>(1);
                } else {
                    prior_data_.reset(new VecT(data_.columns()));
                    (*prior_data_)[0] = static_cast<FT>(1);
                }
                break;
            case GAMMA_BETA:
                if(c == nullptr) throw std::invalid_argument("Can't do gamma_beta with null pointer");
                if constexpr(IsSparseMatrix_v<MatrixType>) {
                    prior_data_.reset(new VecT(data_.columns()));
                    (*prior_data_)[0] = (*c)[0];
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
        row_sums_.resize(data_.rows());
        {
            auto rowsumit = row_sums_.data();
            for(auto r: blz::rowiterator(data_)) {
                if constexpr(blaze::IsDenseMatrix_v<MatrixType>) {
                    if(prior == NONE) {
                        r += 1e-50;
                        assert(blaze::min(r) > 0.);
                    }
                }
                const auto countsum = blaze::sum(r);
                r /= countsum;
                *rowsumit++ = countsum;
            }
        }

        if(blz::detail::needs_logs(measure_)) {
            logdata_.reset(new MatrixType(neginf2zero(log(data_))));
        }
        if(blz::detail::needs_sqrt(measure_)) {
            sqrdata_.reset(new MatrixType(blz::sqrt(data_)));
        }
        if(blz::detail::needs_l2_cache(measure_)) {
            l2norm_cache_.reset(new VecT(data_.rows()));
            OMP_PFOR
            for(size_t i = 0; i < data_.rows(); ++i) {
                l2norm_cache_->operator[](i)  = 1. / blz::l2Norm(weighted_row(i));
            }
        }
        if(blz::detail::needs_probability_l2_cache(measure_)) {
            pl2norm_cache_.reset(new VecT(data_.rows()));
            OMP_PFOR
            for(size_t i = 0; i < data_.rows(); ++i) {
                pl2norm_cache_->operator[](i) = 1. / blz::l2Norm(row(i));
            }
        }
        if(logdata_) {
            jsd_cache_.reset(new VecT(data_.rows()));
            auto &jc = *jsd_cache_;
            for(size_t i = 0; i < jc.size(); ++i) {
                jc[i] = dot(row(i), logrow(i));
            }
        }
    }
    FT get_jsdcache(size_t index) const {
        assert(jsd_cache_ && jsd_cache_->size() > index);
        return (*jsd_cache_)[index];
    }
    FT get_llrcache(size_t index) const {
        assert(jsd_cache_ && jsd_cache_->size() > index);
        return get_jsdcache(index) * row_sums_->operator[](index);
        return (*jsd_cache_)[index] * row_sums_->operator[](index);
    }
}; // DissimilarityApplicator

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

template<typename MJD>
struct BaseOperand {
    using type = decltype(*std::declval<MJD>());
};

template<typename MatrixType, typename PriorContainer=blaze::DynamicVector<typename MatrixType::ElementType, blaze::rowVector>>
auto make_probdiv_applicator(MatrixType &data, DissimilarityMeasure type=JSM, Prior prior=NONE, const PriorContainer *pc=nullptr) {
#if VERBOSE_AF
    std::fprintf(stderr, "[%s:%s:%d] Making probdiv applicator with %d/%s as measure, %d/%s as prior, and %s for prior container.\n",
                 __PRETTY_FUNCTION__, __FILE__, __LINE__, int(type), blz::detail::prob2str(type), int(prior), prior == NONE ? "No prior": prior == DIRICHLET ? "Dirichlet" : prior == GAMMA_BETA ? "Gamma/Beta": "Feature-specific prior",
                pc == nullptr ? "No prior container": (std::string("Container of size ") + std::to_string(pc->size())).data());
#endif
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

template<typename MatrixType, typename WFT=blz::ElementType_t<MatrixType>>
auto make_kmeanspp(const DissimilarityApplicator<MatrixType> &app, unsigned k, uint64_t seed=13, const WFT *weights=nullptr) {
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

} // jsd
using jsd::DissimilarityApplicator;
using jsd::make_d2_coreset_sampler;
using jsd::make_kmc2;
using jsd::make_kmeanspp;
using jsd::make_jsm_applicator;
using jsd::make_probdiv_applicator;



} // minocore

#endif
