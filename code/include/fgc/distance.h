#ifndef FGC_DISTANCE_AND_MEANING_H__
#define FGC_DISTANCE_AND_MEANING_H__
#include "blaze_adaptor.h"
#include <vector>
#include <iostream>

namespace blz {

inline namespace distance {
#define DECL_DIST(norm) \
template<typename FT, bool SO>\
INLINE double norm##Dist(const blaze::DynamicVector<FT, SO> &lhs, const blaze::DynamicVector<FT, SO> &rhs) {\
    return norm##Norm(rhs - lhs);\
}\
template<typename VT, typename VT2, bool SO>\
INLINE double norm##Dist(const blaze::DenseVector<VT, SO> &lhs, const blaze::DenseVector<VT2, SO> &rhs) {\
    return norm##Norm(~rhs - ~lhs);\
}\
\
template<typename VT, typename VT2, bool SO>\
INLINE double norm##Dist(const blaze::DenseVector<VT, SO> &lhs, const blaze::DenseVector<VT2, !SO> &rhs) {\
    return norm##Norm(~rhs - trans(~lhs));\
}\
\
template<typename VT, typename VT2, bool SO>\
INLINE double norm##Dist(const blaze::SparseVector<VT, SO> &lhs, const blaze::SparseVector<VT2, SO> &rhs) {\
    return norm##Norm(~rhs - ~lhs);\
}\
\
template<typename VT, typename VT2, bool SO>\
INLINE double norm##Dist(const blaze::SparseVector<VT, SO> &lhs, const blaze::SparseVector<VT2, !SO> &rhs) {\
    return norm##Norm(~rhs - trans(~lhs));\
}\


DECL_DIST(l1)
DECL_DIST(l2)
DECL_DIST(sqr)
DECL_DIST(l3)
DECL_DIST(l4)
DECL_DIST(max)
DECL_DIST(inf)
#undef DECL_DIST
template<typename FT, typename A, typename OA>
inline double l2Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return l2Dist(CustomVector<const FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<const FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}

template<typename FT, bool SO>
inline double sqrL2Dist(const blz::DynamicVector<FT, SO> &v1, const blz::DynamicVector<FT, SO> &v2) {
    return l2Dist(v1, v2);
}
template<typename FT, blaze::AlignmentFlag AF, blaze::PaddingFlag PF, bool SO, blaze::AlignmentFlag OAF, blaze::PaddingFlag OPF, bool OSO>
inline double sqrL2Dist(const blz::CustomVector<FT, AF, PF, SO> &v1, const blz::CustomVector<FT, OAF, OPF, OSO> &v2) {
    return l2Dist(v1, v2);
}

template<typename FT, typename A, typename OA>
inline double sqrL2Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return sqrL2Dist(CustomVector<const FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                     CustomVector<const FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}

template<typename FT, typename A, typename OA>
INLINE double sqrDist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return sqrL2Dist(lhs, rhs);
}

template<typename FT, typename A, typename OA>
inline double l1Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return l1Dist(CustomVector<const FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<const FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}
template<typename FT, typename A, typename OA>
inline double l3Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return l3Dist(CustomVector<const FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<const FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}

template<typename FT, typename A, typename OA>
inline double l4Dist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return l4Dist(CustomVector<FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}
template<typename FT, typename A, typename OA>
inline double maxDist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {
    return maxDist(CustomVector<FT, blaze::unaligned, blaze::unpadded>(lhs.data(), lhs.size()),
                  CustomVector<FT, blaze::unaligned, blaze::unpadded>(rhs.data(), rhs.size()));
}
template<typename FT, typename A, typename OA>
inline double infDist(const std::vector<FT, A> &lhs, const std::vector<FT, OA> &rhs) {return maxDist(lhs, rhs);}

template<typename Base>
struct sqrBaseNorm: public Base {
    template<typename C1, typename C2>
    INLINE constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        return std::pow(Base::operator()(lhs, rhs), 2);
    }
};
struct L1Norm {
    template<typename C1, typename C2>
    INLINE constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        return l1Dist(lhs, rhs);
    }
};
struct L2Norm {
    template<typename C1, typename C2>
    INLINE constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        return l2Dist(lhs, rhs);
    }
};
struct sqrL2Norm {
    template<typename C1, typename C2>
    INLINE constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        return sqrDist(lhs, rhs);
    }
};
struct L3Norm {
    template<typename C1, typename C2>
    INLINE constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        return l3Dist(lhs, rhs);
    }
};
struct L4Norm {
    template<typename C1, typename C2>
    INLINE constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        return l4Dist(lhs, rhs);
    }
};
struct maxNormFunctor {
    template<typename C1, typename C2>
    INLINE constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        return maxDist(lhs, rhs);
    }
};
struct infNormFunction: maxNormFunctor{};
struct sqrL1Norm: sqrBaseNorm<L1Norm> {};
struct sqrL3Norm: sqrBaseNorm<L3Norm> {};
struct sqrL4Norm: sqrBaseNorm<L4Norm> {};
struct sqrMaxNorm: sqrBaseNorm<maxNormFunctor> {};


// For D^2 sampling.
template<typename BaseDist>
struct SqrNormFunctor: public BaseDist {
    template<typename...Args> SqrNormFunctor(Args &&...args): BaseDist(std::forward<Args>(args)...) {}
    template<typename C1, typename C2>
    INLINE constexpr double operator()(const C1 &lhs, const C2 &rhs) const {
        double basedist = BaseDist::operator()(lhs, rhs);
        return basedist * basedist;
    }
};
template<>
struct SqrNormFunctor<L2Norm>: public sqrL2Norm {};

/*
 *
 * Use https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf
 * to derive KL, Jensen-Shannon divergences + JS distances (sqrt(JS))
 * for the full exponential family of distributions.
 * Currently, what I have is the multinomial, but this also applies to Poisson.
 * See https://en.wikipedia.org/wiki/Exponential_family
 * TODO:
 * These include:
 * Bernoulli, binomial, Poisson, chi-squared, Laplace, normal, lognormal, inverse gaussian, gamma,
 */

template<typename FT, bool SO>
double logsumexp(const blaze::DenseVector<FT, SO> &x) {
    auto maxv = blaze::max(~x);
    return maxv + std::log(blaze::sum(blaze::exp(~x - maxv)));
}

template<typename FT, bool SO>
double logsumexp(const blaze::SparseVector<FT, SO> &x) {
    auto maxv = blaze::max(~x);
    double s = 0.;
    for(const auto p: ~x) {
        s += std::exp(p.value() - maxv); // Sum over sparse elements
    }
    s += ((~x).size() - nonZeros(~x)) * std::exp(-maxv);  // Handle the ones we skipped
    return maxv + std::log(s);
}

template<typename VT1, typename VT2, typename Scalar, bool TF>
double logsumexp(const SVecScalarMultExpr<SVecSVecAddExpr<VT1, VT2, TF>, Scalar, TF> &exp) {
    // Specifically for calculating the logsumexp of the mean of the two sparse vectors.
    // SVecScalarMultExpr doesn't provide a ConstIterator, so we're rolling our own specialized function.
    auto maxv = blaze::max(~exp), mmax = -maxv;
    double s = 0.;
    auto lit = exp.leftOperand().leftOperand().begin(), rit = exp.leftOperand().rightOperand().begin(),
         lie = exp.leftOperand().leftOperand().end(),   rie = exp.leftOperand().rightOperand().end();
    Scalar scalar = exp.rightOperand();
    auto f = [scalar,mmax,&s](auto x) {s += std::exp(std::fma(x, scalar, mmax));};
    size_t nnz = 0;
    for(;;) {
        if(lit == lie) {
            while(rit != rie) {
                f((rit++)->value());
                ++nnz;
            }
            break;
        }
        if(rit == rie) {
            while(lit != lie) {
                f((lit++)->value());
                ++nnz;
            }
            break;
        }
        if(lit->index() < rit->index()) {
            f((lit++)->value());
            ++nnz;
        } else if(rit->index() < lit->index()) {
            f((rit++)->value());
            ++nnz;
        } else /* indexes match */ {
            ++nnz;
            f(lit->value() + rit->value());
            ++lit, ++rit;
        }
    }
    s += ((~exp).size() - nnz) * std::exp(-maxv);  // Handle the ones we skipped
    return maxv + std::log(s);
}

    /* Note: multinomial cumulants for lhs and rhs can be cached as lhc/rhc, such that
    *  multinomial_cumulant(mean) and the dot products are all that are required
    *  TODO: optimize for sparse vectors (maybe filt can be eliminated?)
    *  TODO: cache logs as well as full vectors?

    *  We can turn inf values into 0 because:
    *  Whenever P ( x ) {P(x)} P(x) is zero the contribution of the corresponding term is interpreted as zero because
    *  lim_{x->0+}[xlog(x)] = 0
    *  See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Definition

    *  Alternatively, we can use the Dirichlet prior and avoid any 0s in the first place.
    *  This is done by replacing parameters mu_n = \frac{n}{\sum_{n \in N}{n}})
    *  with                                 mu_n = \frac{n+1}{\sum_{n \in N}{n + 1}}
    *  If this has been done, set filter_nans to be false in any of the downstream functions
    *  See https://mathoverflow.net/questions/72668/how-to-compute-kl-divergence-when-pmf-contains-0s
    *  Another possible prior is the Gamma prior with both parameters $\Beta, \Beta$,
    *  which replaces mu_n with $\frac{n+\Beta}{\sum_{n \in N}{n + \Beta}}$,
    *  as in https://arxiv.org/abs/1202.6201.
    */

template<typename VT>
INLINE double multinomial_cumulant(const VT &x) {return logsumexp(x);}

template<bool VT>
struct FilterNans {
    static constexpr bool value = VT;
};

template<typename FT, bool SO, typename OFT, bool filter_nans=true>
double multinomial_jsd(const blaze::DenseVector<FT, SO> &lhs, const blaze::DenseVector<FT, SO> &rhs, OFT lhc, OFT rhc, [[maybe_unused]] FilterNans<filter_nans> fn=FilterNans<filter_nans>()) {
    DBG_ONLY(if(filter_nans) std::fprintf(stderr, "[%s] Filtering nans\n", __PRETTY_FUNCTION__);)
    using FloatType = typename FT::ElementType;
    auto mean = (~lhs + ~rhs) * .5;
    double retsq, lhv, rhv;
    CONST_IF(filter_nans) {
        NegInf2Zero ni;
        auto logmean = blaze::map(blaze::log(mean), ni);
        auto lhterm = blaze::map(blaze::log(~lhs), ni) - logmean;
        auto rhterm = blaze::map(blaze::log(~rhs), ni) - logmean;
        lhv = dot(lhterm, ~lhs);
        rhv = dot(rhterm, ~rhs);
    } else {
        assert(blaze::min(~lhs) > 0.);
        assert(blaze::min(~rhs) > 0.);
        auto logmean = blaze::log(mean);
        lhv = dot(blaze::log(~lhs) - logmean, ~lhs);
        rhv = dot(blaze::log(~rhs) - logmean, ~rhs);
    }
    retsq = std::max(multinomial_cumulant(mean) + (lhv + rhv - lhc - rhc) * .5, 0.);
#ifndef NDEBUG
    std::fprintf(stderr, "cumulant: %g. lhv: %g. rhv: %g. lhc: %g. rhc: %g. retsq: %g\n", multinomial_cumulant(mean), lhv, rhv, lhc, rhc, retsq);
#endif
    return std::sqrt(retsq);
}

template<typename VT, typename VT2, bool SO, typename OFT, typename OBufType>
double multinomial_jsd(const blaze::DenseVector<VT, SO> &lhs,
                       const blaze::DenseVector<VT, SO> &rhs,
                       const blaze::DenseVector<VT2, SO> &lhlog,
                       const blaze::DenseVector<VT2, SO> &rhlog,
                       OFT lhc, OFT rhc,
                       OBufType &meanbuf,
                       OBufType &logmeanbuf)
{
    assert(blaze::min(rhs) > 0. || !std::fprintf(stderr, "This version of the function requires 0s be removed\n"));
    ~(*meanbuf) = (~lhs + ~rhs) * .5;
    ~*logmeanbuf = blaze::map(blaze::log(~*meanbuf), blaze::NegInf2Zero());
    return multinomial_cumulant(~*meanbuf) +
        (0.5 * (dot(lhlog - ~*logmeanbuf, ~lhs) +
          + dot(rhlog - ~*logmeanbuf, ~rhs) - lhc - rhc));
}


template<typename FT, typename VT2, bool SO, typename OFT, bool filter_nans=true>
double multinomial_jsd(const blaze::DenseVector<FT, SO> &lhs, const blaze::DenseVector<FT, SO> &rhs,
                       // Original vectors
                       const blaze::DenseVector<VT2, SO> &lhl, const blaze::DenseVector<VT2, SO> &rhl,
                       // cached logs
                       OFT lhc, OFT rhc,
                       [[maybe_unused]] FilterNans<filter_nans> fn=FilterNans<filter_nans>())
{
    std::cout << "lhl: \n" << lhl << '\n';
    DBG_ONLY(if(filter_nans) std::fprintf(stderr, "[%s] Filtering nans\n", __PRETTY_FUNCTION__);)
    using FloatType = typename FT::ElementType;
    assert(std::find_if((~lhl).begin(), (~lhl).end(), [](auto x)
                        {return std::isinf(x) || std::isnan(x);})
           == (~lhl).end());
    assert(std::find_if((~rhl).begin(), (~rhl).end(), [](auto x)
                        {return std::isinf(x) || std::isnan(x);})
           == (~rhl).end());
    auto mean = (~lhs + ~rhs) * .5;
    double lhv, rhv;
    CONST_IF(filter_nans) {
        auto logmean = blaze::map(blaze::log(mean), NegInf2Zero());
        lhv = dot(lhl - logmean, ~lhs);
        rhv = dot(rhl - logmean, ~rhs);
    } else {
        auto logmean = blaze::log(mean);
        lhv = dot(lhl - logmean, ~lhs);
        rhv = dot(rhl - logmean, ~rhs);
    }
    return std::sqrt(multinomial_cumulant(mean) + .5 * (lhv + rhv - lhc - rhc));
}

template<typename FT, bool SO, typename LT, typename OFT>
double multinomial_jsd(const blaze::SparseVector<FT, SO> &lhs, const blaze::SparseVector<FT, SO> &rhs,
                       // Original vectors
                       const LT &lhl, const LT &rhl,
                       //const blaze::SparseVector<FT, SO> &lhl, const blaze::SparseVector<FT, SO> &rhl,
                       // cached logs
                       OFT lhc, OFT rhc) {
    assert(std::find_if(lhl.begin(), lhl.end(), [](auto x)
                        {return std::isinf(x.value()) || std::isnan(x.value());})
           == lhl.end());
    assert(std::find_if(rhl.begin(), rhl.end(), [](auto x)
                        {return std::isinf(x.value()) || std::isnan(x.value());})
           == rhl.end());
    auto mean = (~lhs + ~rhs) * .5;
    auto logmean = blaze::log(mean);
    double lhv = dot(lhl - logmean, ~lhs);
    double rhv = dot(rhl - logmean, ~rhs);
    double retsq = std::max(multinomial_cumulant(mean) +  .5 * (lhv + rhv - lhc - rhc), 0.);
#ifndef NDEBUG
    std::fprintf(stderr, "cumulant: %g. lhv: %g. rhv: %g. lhc: %g. rhc: %g. retsq: %g\n", multinomial_cumulant(mean), lhv, rhv, lhc, rhc, retsq);
#endif
    return std::sqrt(retsq);
}

template<typename FT, bool SO>
double multinomial_jsd(const blaze::SparseVector<FT, SO> &lhs, const blaze::SparseVector<FT, SO> &rhs) {
    auto lhl = blaze::log(lhs);
    auto rhl = blaze::log(rhs);
    return multinomial_jsd(lhs, rhs, lhl, rhl, multinomial_cumulant(lhs), multinomial_cumulant(rhs));
}

template<typename FT, bool SO, bool filter_nans=true>
INLINE double multinomial_jsd(const blaze::DenseVector<FT, SO> &lhs,
                              const blaze::DenseVector<FT, SO> &rhs,
                              FilterNans<filter_nans> fn=FilterNans<filter_nans>())
{
    assert((~lhs).size() == (~rhs).size());
    return multinomial_jsd(lhs, rhs, multinomial_cumulant(~lhs), multinomial_cumulant(~rhs), fn);
}

enum Prior {
    NONE,       // Do not modify values. Requires that there are no nonzero parameters
    DIRICHLET,  // Uniform smoothing
    GAMMA_BETA, // Requires a parameter
    FEATURE_SPECIFIC_PRIOR // Requires a vector of parameters
};

template<typename DenseType>
class MultinomialJSDApplicator {
    using FT = typename DenseType::ElementType;

    //using opposite_type = typename base_type::OppositeType;
    DenseType &data_;
    blaze::DynamicVector<FT> cached_cumulants_;
    std::unique_ptr<blaze::DynamicMatrix<FT>> logdata_;
public:
    const Prior prior_;
    template<typename PriorContainer=blaze::DynamicVector<FT, blaze::rowVector>>
    MultinomialJSDApplicator(DenseType &ref,
                             Prior prior=DIRICHLET,
                             const PriorContainer *c=nullptr):
        data_(ref), logdata_(nullptr), prior_(prior)
    {
        prep(c);
    }
    void set_logs(DenseType &ref) {
        if(ref.rows() = data_.rows() || ref.columns() != data_.columns())
            throw std::invalid_argument("set_logs requires a matrix of the same dimensions");
        logdata_ = &ref;
        ref = blaze::map(blaze::log(data_), NegInf2Zero());
    }
    double operator()(size_t lhind, size_t rhind) const {
        return jsd(lhind, rhind);
    }
    double jsd(size_t lhind, size_t rhind) const {
        assert(lhind < cached_cumulants_.size());
        assert(rhind < cached_cumulants_.size());
        const auto lhv = cached_cumulants_[lhind],
                   rhv = cached_cumulants_[rhind];
        if(logdata_) {
            assert(logdata_->rows() == data_.rows());
            return multinomial_jsd(row(data_, lhind BLAZE_CHECK_DEBUG),
                                   row(data_, rhind BLAZE_CHECK_DEBUG),
                                   row(*logdata_, lhind BLAZE_CHECK_DEBUG),
                                   row(*logdata_, rhind BLAZE_CHECK_DEBUG),
                                   lhv,
                                   rhv);
        } else {
            return multinomial_jsd(row(data_, lhind BLAZE_CHECK_DEBUG),
                                   row(data_, rhind BLAZE_CHECK_DEBUG),
                                   lhv,
                                   rhv);
        }
    }
private:
    template<typename Container=blaze::DynamicVector<FT, blaze::rowVector>>
    void prep(const Container *c=nullptr) {
        switch(prior_) {
            case NONE:
            assert(min(data_) > 0.);
            return;
            case DIRICHLET: data_ += (1. / data_.columns()); break;
            case GAMMA_BETA:
                if(c == nullptr) throw std::invalid_argument("Can't do gamma_beta with null pointer");
                 data_ += (1. / *std::begin(*c)); break;
            case FEATURE_SPECIFIC_PRIOR:
                if(c == nullptr) throw std::invalid_argument("Can't do feature-specific with null pointer");
                for(auto rw: blz::rowiterator(data_))
                    rw += *c;
        }
        for(size_t i = 0; i < data_.rows(); ++i)
            row(data_, i) /= blaze::l2Norm(row(data_, i));
        logdata_.reset(new blaze::DynamicMatrix<FT>(data_.rows(), data_.columns()));
        *logdata_ = map(map(data_, Log()), NegInf2Zero());
        std::cout << row(*logdata_, 0);
        cached_cumulants_.resize(data_.rows());
        for(size_t i = 0; i < data_.rows(); ++i)
            cached_cumulants_[i] = multinomial_cumulant(row(data_, i BLAZE_CHECK_DEBUG));
    }
};

template<typename FT, bool SO>
INLINE double multinomial_bregman(const blaze::DenseVector<FT, SO> &lhs,
                                  const blaze::DenseVector<FT, SO> &rhs,
                                  const blaze::DenseVector<FT, SO> &lhlog,
                                  const blaze::DenseVector<FT, SO> &rhlog)
{
    assert_all_nonzero(lhs);
    assert_all_nonzero(rhs);
    return blaze::dot(lhs, lhlog - rhlog);
}
template<typename FT, bool SO>
INLINE double      poisson_bregman(const blaze::DenseVector<FT, SO> &lhs,
                                   const blaze::DenseVector<FT, SO> &rhs,
                                   const blaze::DenseVector<FT, SO> &lhlog,
                                   const blaze::DenseVector<FT, SO> &rhlog)
{
    // Assuming these are all independent (not ideal)
    assert_all_nonzero(lhs);
    assert_all_nonzero(rhs);
    return blaze::dot(lhs, lhlog - rhlog) + blaze::sum(rhs - lhs);
}

template<typename LHVec, typename RHVec>
double bhattacharya_measure(const LHVec &lhs, const RHVec &rhs) {
    // Requires same storage.
    // TODO: generalize for different storage classes/transpose flags using DenseVector and SparseVector
    // base classes
    return sqrt(lhs * rhs);
}

template<typename LHVec, typename RHVec>
double bhattacharya_metric(const LHVec &lhs, const RHVec &rhs) {
    // Comaniciu, D., Ramesh, V. & Meer, P. (2003). Kernel-based object tracking.IEEE Transactionson Pattern Analysis and Machine Intelligence,25(5), 564-577.
    // Proves that this extension is a valid metric
    // See http://www.cse.yorku.ca/~kosta/CompVis_Notes/bhattacharyya.pdf
    return std::sqrt(1. - bhattacharya_measure(lhs, rhs));
}
template<typename LHVec, typename RHVec>
double matusita_distance(const LHVec &lhs, const RHVec &rhs) {
    return sqrL2Dist(sqrt(lhs), sqrt(rhs));
}
} // distance

} // namespace blz

#endif // FGC_DISTANCE_AND_MEANING_H__
