#pragma once
#ifndef FGC_KMEANS_H__
#define FGC_KMEANS_H__
#include <cassert>
#include <mutex>
#include <numeric>
#include "matrix_coreset.h"
#include "timer.h"

namespace fgc {
using blz::rowiterator;



struct MatrixLookup {};
template<typename WFT>
struct WeightedMatrixLookup {
    const WFT *w_;
    WeightedMatrixLookup(const WFT *ptr=nullptr): w_(ptr) {}
};

template<typename Mat>
struct MatrixMetric {
    /*
     *  This calculate the distance between item i and item j in this problem
     *  by simply indexing the given array.
     *  This requires precalculation of the array (and space) but saves computation.
     *  By convention, use row index = facility, column index = point
     */
    const Mat &mat_;
    MatrixMetric(const Mat &mat): mat_(mat) {}
    auto operator()(size_t i, size_t j) const {
#if VERBOSE_AF
        std::fprintf(stderr, "Row %zu and column %zu have value %f\n", i, j, double(mat_(i, j)));
#endif
        return mat_(i, j);
    }
};

template<typename Mat, typename WFT>
struct WeightedMatrixMetric: MatrixMetric<Mat> {
    /*
     * Weighted version of super-class
     * By convention, use row index = facility, column index = point
     * For this reason, we index weights under index j
     */
    using super = MatrixMetric<Mat>;
    const WFT *weights_;

    WeightedMatrixMetric(const Mat &mat, const WFT *weights=nullptr): weights_(weights) {}
    INLINE auto mul(size_t ind) const {return weights_ ? weights_[ind]: static_cast<WFT>(1.);}
    auto operator()(size_t i, size_t j) const {
        return super::operator()(i, j) * mul(j);
    }
};

template<typename Mat, typename Dist>
struct MatrixDistMetric {
    /*
     *  This calculate the distance between item i and item j in this problem
     *  by calculating the distances between row i and row j under the given distance metric.
     *  This requires precalculation of the array (and space) but saves computation.
     *
     */
    const Mat &mat_;
    const Dist dist_;

    MatrixDistMetric(const Mat &mat, Dist dist): mat_(mat), dist_(std::move(dist)) {}

    auto operator()(size_t i, size_t j) const {
        return dist_(row(mat_, i, blaze::unchecked), row(mat_, j, blaze::unchecked));
    }
};
template<typename Iter, typename Dist>
struct IndexDistMetric {
    /*
     * Adapts random access iterator to use norms between dereferenced quantities.
     */
    const Iter iter_;
    const Dist &dist_;

    IndexDistMetric(const Iter iter, const Dist &dist): iter_(iter), dist_(std::move(dist)) {}

    auto operator()(size_t i, size_t j) const {
        return dist_(iter_[i], iter_[j]);
    }
};

template<typename Iter>
struct BaseOperand {
    using DerefType = decltype((*std::declval<Iter>()));
    using TwiceDerefedType = std::remove_reference_t<decltype(std::declval<DerefType>().operand())>;
    using type = TwiceDerefedType;
};


template<typename Iter>
struct IndexDistMetric<Iter, MatrixLookup> {
    using Operand = typename BaseOperand<Iter>::type;
    using ET = typename Operand::ElementType;
    /* Specialization of above for MatrixLookup
     *
     *
     */
    using Dist = MatrixLookup;
    const Operand &mat_;
    const Dist dist_;
    //TD<Operand> to2;

    IndexDistMetric(const Iter iter, Dist dist): mat_((*iter).operand()), dist_(std::move(dist)) {}

    ET operator()(size_t i, size_t j) const {
        return mat_(i, j);
        //return iter_[i][j];
    }
};



#if 0
template<typename MatrixType>
struct IndexDistMetric<MatrixType, jsd::MultinomialJSDApplicator<MatrixType>> {
    using ET = typename MatrixType::ElementType;
    using App = typename jsd::MultinomialJSDApplicator<MatrixType>;
    /* Specialization of above for MatrixLookup
     *
     *
     */
    const App &app_;
    //TD<Operand> to2;

    IndexDistMetric(const MatrixType &mat, const App &app): app_(app) {}

    ET operator()(size_t i, size_t j, bool use_jsm=true) const {
        assert(i < app_.size());
        assert(j < app_.size());
        return app_(i, j);
    }
};
#endif

template<typename Iter, typename Dist>
auto make_index_dm(const Iter iter, const Dist &dist) {
    return IndexDistMetric<Iter, Dist>(iter, dist);
}
template<typename Mat, typename Dist>
auto make_matrix_dm(const Mat &mat, const Dist &dist) {
    return MatrixDistMetric<Mat, Dist>(mat, dist);
}
template<typename Mat>
auto make_matrix_m(const Mat &mat) {
    return MatrixMetric<Mat>(mat);
}

namespace coresets {



using std::partial_sum;
using blz::sqrL2Norm;

template<typename C>
using ContainedTypeFromIterator = std::decay_t<decltype((*std::declval<C>())[0])>;


/*
 * However, using https://arxiv.org/abs/1508.05243 (Strong Coresets for Hard and Soft Bregman Clustering withApplications to Exponential Family Mixtures)
 * any squared Bregman divergence will work for the kmeanspp, including regular exponential families.
 * See http://www.jmlr.org/papers/volume6/banerjee05b/banerjee05b.pdf
 * http://www.cs.utexas.edu/users/inderjit/Talks/bregtut.pdf
 * and perhaps https://arxiv.org/pdf/1309.7109.pdf.
 * The Banerjee paper has a table of relevant information.
 */

template<typename Oracle, typename FT=float,
         typename IT=std::uint32_t, typename RNG>
std::tuple<std::vector<IT>, std::vector<IT>, std::vector<FT>>
kmeanspp(const Oracle &oracle, RNG &rng, size_t np, size_t k) {
    std::vector<IT> centers;
    std::vector<FT> distances(np, 0.), cdf(np);
    double sumd2 = 0.;
    {
        auto fc = rng() % np;
        centers.push_back(fc);
#ifdef _OPENMP
        OMP_PRAGMA("omp parallel for reduction(+:sumd2)")
#else
        SK_UNROLL_8
#endif
        for(size_t i = 0; i < np; ++i) {
            if(unlikely(i == fc)) continue;
            double dist = oracle(fc, i);
            distances[i] = dist;
            sumd2 += dist;
        }
        assert(distances[fc] == 0.);
        inclusive_scan(distances.begin(), distances.end(), cdf.begin());
    }
#if VERBOSE_AF
    std::fprintf(stderr, "first loop sum: %f. manual: %f\n", sumd2, std::accumulate(distances.begin(), distances.end(), double(0)));
#endif
    std::vector<IT> assignments(np);
    std::uniform_real_distribution<double> urd;
    while(centers.size() < k) {
        // At this point, the cdf has been prepared, and we are ready to sample.
        // add new element
        auto newc = std::lower_bound(cdf.begin(), cdf.end(), cdf.back() * urd(rng)) - cdf.begin();
        const auto current_center_id = centers.size();
        centers.push_back(newc);
        sumd2 -= distances[newc];
        distances[newc] = 0.;
        double sum = sumd2;
        OMP_PRAGMA("omp parallel for reduction(+:sum)")
        for(IT i = 0; i < np; ++i) {
            if(unlikely(i == newc)) continue;
            auto &ldist = distances[i];
            double dist = oracle(newc, i);
            if(dist < ldist) { // Only write if it changed
                assignments[i] = current_center_id;
                auto diff = dist - ldist;
                sum += diff;
                ldist = dist;
            }
        }
        sumd2 = sum;
#if VERBOSE_AF
        std::fprintf(stderr, "sumd2: %f. manual: %f\n", sumd2, std::accumulate(distances.begin(), distances.end(), double(0)));
#endif
        inclusive_scan(distances.begin(), distances.end(), cdf.begin());
    }
    return std::make_tuple(std::move(centers), std::move(assignments), std::move(distances));
}

template<typename Iter, typename FT=ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG, typename Norm=sqrL2Norm>
std::tuple<std::vector<IT>, std::vector<IT>, std::vector<FT>>
kmeanspp(Iter first, Iter end, RNG &rng, size_t k, const Norm &norm=Norm()) {
    auto dm = make_index_dm(first, norm);
    static_assert(std::is_floating_point<FT>::value, "FT must be fp");
    return kmeanspp(dm, rng, end - first, k);
}

template<typename Oracle, typename Sol, typename FT=float, typename IT=uint32_t>
std::pair<blz::DV<IT>, blz::DV<FT>> get_oracle_costs(const Oracle &oracle, size_t np, const Sol &sol)
{
    blz::DV<IT> assignments(np);
    blz::DV<FT> costs(np, std::numeric_limits<FT>::max());
    util::Timer t("get oracle costs");
    OMP_PFOR
    for(size_t i = 0; i < np; ++i) {
        auto it = sol.begin();
        FT mincost = oracle(*it, i);
        IT minind = 0, cind = 0;
        while(++it != sol.end()) {
            if(FT newcost = oracle(*it, i); newcost < mincost)
                mincost = newcost, minind = cind;
            ++cind;
        }
        costs[i] = mincost;
        assignments[i] = minind;
    }
    std::fprintf(stderr, "Centers have total cost %g\n", blz::sum(costs));
    return std::make_pair(assignments, costs);
}

/*
 * Implementation of the $KMC^2$ algorithm from:
 * Bachem, et al. Approximate K-Means++ in Sublinear Time (2016)
 * Available at https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12147/11759
 */

template<typename Oracle,
         typename IT=std::uint32_t, typename RNG>
std::vector<IT>
kmc2(const Oracle &oracle, RNG &rng, size_t np, size_t k, size_t m = 2000)
{
    if(m == 0) throw std::invalid_argument("m must be nonzero");
    flat_hash_set<IT> centers{IT(rng() % np)};
    // Helper function for minimum distance
    auto mindist = [&centers,&oracle](auto newind) {
        typename flat_hash_set<IT>::const_iterator it = centers.begin(), end = centers.end();
        auto dist = oracle(*it, newind);
        while(++it != end)
            dist = std::min(dist, oracle(*it, newind));
        return dist;
    };

    while(centers.size() < k) {
        auto x = rng() % np;
        auto xdist = mindist(x);
        auto baseseed = rng();
        OMP_PFOR
        for(unsigned j = 1; j < m; ++j) {
            uint64_t local_seed = baseseed + j;
            wy::wyhash64_stateless(&local_seed);
            auto y = local_seed % np;
            auto ydist = mindist(y);
            auto rat = ydist / xdist;
            wy::wyhash64_stateless(&local_seed);
            auto urd_val = double(local_seed) / std::numeric_limits<uint64_t>::max();
            if(rat > urd_val) {
                OMP_CRITICAL
                {
                    if(rat > urd_val)
                        x = y, xdist = ydist;
                }
            }
        }
        centers.insert(x);
    }
    return std::vector<IT>(centers.begin(), centers.end());
}
template<typename Iter, typename FT=ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG, typename Norm=sqrL2Norm>
std::vector<IT>
kmc2(Iter first, Iter end, RNG &rng, size_t k, size_t m = 2000, const Norm &norm=Norm()) {
    if(m == 0) throw std::invalid_argument("m must be nonzero");
    auto dm = make_index_dm(first, norm);
    static_assert(std::is_floating_point<FT>::value, "FT must be fp");
    size_t np = end - first;
    return kmc2(dm, rng, np, k, m);
}


template<typename MT, bool SO,
         typename IT=std::uint32_t, typename RNG, typename Norm=sqrL2Norm>
auto
kmeanspp(const blaze::Matrix<MT, SO> &mat, RNG &rng, size_t k, const Norm &norm=Norm(), bool rowwise=true) {
    using FT = typename MT::ElementType;
    std::tuple<std::vector<IT>, std::vector<IT>, std::vector<FT>> ret;
    if(rowwise) {
        auto rowit = blz::rowiterator(~mat);
        ret = kmeanspp(rowit.begin(), rowit.end(), rng, k, norm);
    } else { // columnwise
        auto columnit = blz::columniterator(~mat);
        ret = kmeanspp(columnit.begin(), columnit.end(), rng, k, norm);
    }
    return ret;
}

template<typename MT, bool SO,
         typename IT=std::uint32_t, typename RNG, typename Norm=sqrL2Norm>
auto
kmc2(const blaze::Matrix<MT, SO> &mat, RNG &rng, size_t k,
     size_t m=2000,
     const Norm &norm=Norm(), 
     bool rowwise=true)
{
    std::vector<IT> ret;
    if(rowwise) {
        auto rowit = blz::rowiterator(~mat);
        ret = kmc2(rowit.begin(), rowit.end(), rng, k, m, norm);
    } else { // columnwise
        auto columnit = blz::columniterator(~mat);
        ret = kmc2(columnit.begin(), columnit.end(), rng, k, m, norm);
    }
    return ret;
}

template<typename IT, typename MatrixType, typename CMatrixType=MatrixType, typename WFT=double>
double lloyd_iteration(std::vector<IT> &assignments, std::vector<WFT> &counts,
                     CMatrixType &centers, MatrixType &data,
                     const WFT *weights=nullptr)
{
    static_assert(std::is_floating_point_v<WFT>, "WTF must be floating point for weighted kmeans");
    // make sure this is only rowwise/rowMajor
    assert(counts.size() == centers.rows());
    assert(centers.columns() == data.columns());
    // 1. Gets means of assignments
    centers = static_cast<typename CMatrixType::ElementType_t>(0.);
    const size_t nc = centers.columns(), nr = data.rows();
    auto getw = [weights](size_t ind) {
        return weights ? weights[ind]: WFT(1.);
    };
    // TODO: parallelize (one thread per center, maybe?)
    for(size_t i = 0; i < nr; ++i) {
        assert(assignments[i] < centers.size());
        auto asn = assignments[i];
        auto dr = row(data, i BLAZE_CHECK_DEBUG);
        auto cr = row(centers, asn BLAZE_CHECK_DEBUG);
        const auto w = getw(i);
        cr += (dr * w);
        counts[asn] += w;
    }
    for(size_t i = 0; i < centers.rows(); ++i) {
        assert(counts[i]);
        row(centers, i BLAZE_CHECK_DEBUG) /= counts[i];
    }
    // 2. Assign centers
    double total_loss = 0.;
    OMP_PRAGMA("omp parallel for reduction(+:total_loss)")
    for(size_t i = 0; i < nr; ++i) {
        auto dr = row(data, i BLAZE_CHECK_DEBUG);
        auto dist = std::numeric_limits<double>::max();
        double newdist;
        unsigned label = -1;
        for(size_t j = 0; j < centers.rows(); ++j) {
            if((newdist = sqrL2Dist(dr, row(centers, i BLAZE_CHECK_DEBUG))) < dist) {
                dist = newdist;
                label = j;
            }
        }
        assignments[i] = label;
        total_loss += getw(i) * dist;
    }
    return total_loss;
}

template<typename IT, typename MatrixType, typename CMatrixType=MatrixType, typename WFT=double>
void lloyd_loop(std::vector<IT> &assignments, std::vector<WFT> &counts,
                     CMatrixType &centers, MatrixType &data,
                     double tolerance=0., size_t maxiter=-1,
                     const WFT *weights=nullptr)
{
    if(tolerance < 0.) throw 1;
    size_t iternum = 0;
    double oldloss = std::numeric_limits<double>::max(), newloss;
    for(;;) {
        newloss = lloyd_iteration(assignments, counts, centers, data, weights);
        if(iternum++ == maxiter || std::abs(oldloss - newloss) < tolerance)
            break;
        std::fprintf(stderr, "new loss at %zu: %g. old loss: %g\n", iternum, newloss, oldloss);
        oldloss = newloss;
    }
    for(;;) {
        double newloss = lloyd_iteration(assignments, counts, centers, data, weights);
        if(std::abs(oldloss - newloss) / oldloss < tolerance || iternum++ == maxiter) return;
        oldloss = newloss;
    }
}



template<typename Iter,
         typename IT=std::uint32_t, typename RNG=wy::WyRand<uint32_t, 2>,
         typename FT=ContainedTypeFromIterator<Iter>>
auto kmeans_coreset(Iter start, Iter end,
                    size_t k, RNG &rng,
                    size_t cs_size,
                    const FT *weights=nullptr) {
    auto [centers, assignments, sqdists] = kmeanspp(start, end, rng, k, sqrL2Norm());
    using sq_t = typename decltype(sqdists)::value_type;
    coresets::CoresetSampler<sq_t, IT> cs;
    size_t np = end - start;
    cs.make_sampler(np, centers.size(), sqdists.data(), assignments.data(), weights,
                    /*seed=*/rng());
    auto ics(cs.sample(cs_size, rng()));
    static_assert(std::is_same<decltype(ics), coresets::IndexCoreset<IT, sq_t>>::value, "must be this type");
    //coresets::IndexCoreset<IT, sq_t> ics(cs.sample(cs_size, rng()));
#ifndef NDEBUG
    std::fprintf(stderr, "max sampled idx: %u\n", *std::max_element(ics.indices_.begin(), ics.indices_.end()));
#endif
    return ics;
}
template<typename FT, bool SO,
         typename IT=std::uint32_t, typename RNG=wy::WyRand<uint32_t, 2>>
auto kmeans_matrix_coreset(const blaze::DynamicMatrix<FT, SO> &mat, size_t k, RNG &rng, size_t cs_size,
                           const FT *weights=nullptr, bool rowwise=true)
{
    if(!rowwise) throw std::runtime_error("Not implemented");
    const auto &blzview = reinterpret_cast<const blz::DynamicMatrix<FT, SO> &>(mat);
    auto ics = kmeans_coreset(blzview.rowiterator().begin(), blzview.rowiterator().end(),
                              k, rng, cs_size, weights);
#ifndef NDEBUG
    std::fprintf(stderr, "Got kmeans coreset of size %zu\n", ics.size());
#endif
    coresets::MatrixCoreset<blaze::DynamicMatrix<FT, SO>, FT> csmat = index2matrix(ics, mat);
    return csmat;
}

// TODO: 1. get run kmeans clustering on MatrixCoreset
//       2. Use this for better coreset construction (since coreset size is dependent on the approximation ratio)
//       3. Generate new solution
//       4. Iterate over this
//       5. ???
//       6. PROFIT
//       Why?
//       The solution is 1 + eps accurate, with the error being 1/eps^2
//       We can effectively remove the log(n) approximation
//       ratio from
//       Epilogue.


} // namespace coresets
} // namespace fgc
#endif // FGC_KMEANS_H__
