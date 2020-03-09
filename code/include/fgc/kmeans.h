#pragma once
#ifndef FGC_KMEANS_H__
#define FGC_KMEANS_H__
#include <cassert>
#include <mutex>
#include <numeric>
#include "matrix_coreset.h"
#include "timer.h"
#include "div.h"

namespace fgc {
using blz::rowiterator;



struct MatrixLookup {};

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
        return mat_(i, j);
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
using blz::distance::sqrL2Norm;

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

template<typename Oracle, typename FT=std::decay_t<decltype(std::declval<Oracle>()(0,0))>,
         typename IT=std::uint32_t, typename RNG, typename WFT=FT>
std::tuple<std::vector<IT>, std::vector<IT>, std::vector<FT>>
kmeanspp(const Oracle &oracle, RNG &rng, size_t np, size_t k, const WFT *weights=nullptr) {
    std::vector<IT> centers;
    std::vector<FT> distances(np, 0.), cdf(np);
    {
        auto fc = rng() % np;
        centers.push_back(fc);
#ifdef _OPENMP
        OMP_PFOR
#else
        SK_UNROLL_8
#endif
        for(size_t i = 0; i < np; ++i) {
            if(unlikely(i == fc)) continue;
            double dist = oracle(fc, i);
            distances[i] = dist;
        }
        assert(distances[fc] == 0.);
        if(weights) inclusive_scan(distances.begin(), distances.end(), cdf.begin(), [weights,ds=&distances[0]](auto x, const auto &y) {
            return x + y * weights[&y - ds];
        });
        else inclusive_scan(distances.begin(), distances.end(), cdf.begin());
    }
    std::vector<IT> assignments(np);
    std::uniform_real_distribution<double> urd;
    while(centers.size() < k) {
        // At this point, the cdf has been prepared, and we are ready to sample.
        // add new element
        IT newc;
        do {
            newc = std::lower_bound(cdf.begin(), cdf.end(), cdf.back() * urd(rng)) - cdf.begin();
        } while(newc >= np);
        const auto current_center_id = centers.size();
        assignments[newc] = centers.size();
        centers.push_back(newc);
        distances[newc] = 0.;
        OMP_PFOR
        for(IT i = 0; i < np; ++i) {
            if(unlikely(i == newc)) continue;
            auto &ldist = distances[i];
            auto dist = oracle(newc, i);
            if(dist < ldist) { // Only write if it changed
                assignments[i] = current_center_id;
                ldist = dist;
            }
        }
        if(weights) inclusive_scan(distances.begin(), distances.end(), cdf.begin(), [weights,ds=&distances[0]](auto x, const auto &y) {
            return x + y * weights[&y - ds];
        });
        else inclusive_scan(distances.begin(), distances.end(), cdf.begin());
    }
    return std::make_tuple(std::move(centers), std::move(assignments), std::move(distances));
}

template<typename Iter, typename FT=ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG, typename Norm=sqrL2Norm, typename WFT=FT>
std::tuple<std::vector<IT>, std::vector<IT>, std::vector<FT>>
kmeanspp(Iter first, Iter end, RNG &rng, size_t k, const Norm &norm=Norm(), WFT *weights=nullptr) {
    auto dm = make_index_dm(first, norm);
    static_assert(std::is_floating_point<FT>::value, "FT must be fp");
    return kmeanspp<decltype(dm), FT>(dm, rng, end - first, k, weights);
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
        auto mincost = oracle(*it, i);
        IT minind = 0, cind = 0;
        while(++it != sol.end()) {
            if(auto newcost = oracle(*it, i); newcost < mincost)
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
    schism::Schismatic<IT> div(np);
    flat_hash_set<IT> centers{div.mod(rng())};
    // Helper function for minimum distance
    auto mindist = [&centers,&oracle](auto newind) {
        typename flat_hash_set<IT>::const_iterator it = centers.begin(), end = centers.end();
        auto dist = oracle(*it, newind);
        while(++it != end)
            dist = std::min(dist, oracle(*it, newind));
        return dist;
    };

    while(centers.size() < k) {
        auto x = div.mod(rng());
        double xdist = oracle(*centers.begin(), x);
        OMP_PRAGMA("omp parallel for reduction(min:x)")
        for(unsigned i = 1; i < centers.size(); ++i) {
            auto it = centers.begin();
            std::advance(it, i);
            xdist = std::min(xdist, oracle(*it, x));
        }
        auto xdist = mindist(x);
        auto xdi = 1. / xdist;
        auto baseseed = rng();
        const double max64inv = 1. / std::numeric_limits<uint64_t>::max();
        OMP_PFOR
        for(unsigned j = 1; j < m; ++j) {
            uint64_t local_seed = baseseed + j;
            wy::wyhash64_stateless(&local_seed);
            auto y = div.mod(local_seed);
            auto ydist = mindist(y);
            wy::wyhash64_stateless(&local_seed);
            const auto urd_val = local_seed * max64inv;
            if(ydist * xdi > urd_val) {
                OMP_CRITICAL
                {
                    if(ydist * xdi > urd_val)
                        x = y, xdist = ydist, xdi = 1. / xdist;
                }
            }
        }
        std::fprintf(stderr, "[kmc2]: %zu/%zu\n", centers.size(), size_t(k));
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
         typename IT=std::uint32_t, typename RNG, typename Norm=sqrL2Norm, typename WFT=typename MT::ElementType>
auto
kmeanspp(const blaze::Matrix<MT, SO> &mat, RNG &rng, size_t k, const Norm &norm=Norm(), bool rowwise=true, const WFT *weights=nullptr) {
    using FT = typename MT::ElementType;
    std::tuple<std::vector<IT>, std::vector<IT>, std::vector<FT>> ret;
    if(rowwise) {
        auto rowit = blz::rowiterator(~mat);
        ret = kmeanspp(rowit.begin(), rowit.end(), rng, k, norm, weights);
    } else { // columnwise
        auto columnit = blz::columniterator(~mat);
        ret = kmeanspp(columnit.begin(), columnit.end(), rng, k, norm, weights);
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

#define DUMBAVE 1

template<typename IT, typename MatrixType, typename CMatrixType=MatrixType, typename WFT=double, typename Functor=blz::sqrL2Norm>
double lloyd_iteration(std::vector<IT> &assignments, std::vector<WFT> &counts,
                       CMatrixType &centers, MatrixType &data,
                       const Functor &func=Functor(),
                       const WFT *weights=nullptr)
{
    static_assert(std::is_floating_point_v<WFT>, "WTF must be floating point for weighted kmeans");
    // make sure this is only rowwise/rowMajor
    assert(counts.size() == centers.rows() || !std::fprintf(stderr, "counts size: %zu. centers rows: %zu\n", counts.size(), centers.rows()));
    assert(centers.columns() == data.columns());
    // 1. Gets means of assignments
    const size_t nr = data.rows();
    auto getw = [weights](size_t ind) {
        return weights ? weights[ind]: WFT(1.);
    };
    OMP_ONLY(std::unique_ptr<std::mutex[]> mutexes = std::make_unique<std::mutex[]>(centers.rows());)
    centers = static_cast<typename CMatrixType::ElementType>(0.);
    std::memset(counts.data(), 0, counts.size() * sizeof(counts[0]));
    assert(blz::sum(centers) == 0.);
    bool centers_reassigned;
    std::unique_ptr<typename MatrixType::ElementType[]> costs;
    get_assignment_counts:
    centers_reassigned = false;
    
    OMP_PRAGMA("omp parallel for schedule(dynamic)")
    for(size_t i = 0; i < nr; ++i) {
        assert(assignments[i] < centers.rows());
        auto asn = assignments[i];
        auto dr = row(data, i BLAZE_CHECK_DEBUG);
        auto cr = row(centers, asn BLAZE_CHECK_DEBUG);
        const auto w = getw(i);
        {
            OMP_ONLY(std::lock_guard<std::mutex> lg(mutexes[asn]);)
            if(w == 1.) {
                blz::serial(cr.operator+=(dr));
            }
            else blz::serial(cr.operator+=(dr * w));
        }
        OMP_ATOMIC
        counts[asn] += w;
    }
    std::fprintf(stderr, "Assigned cluster centers\n");
    for(size_t i = 0; i < centers.rows(); ++i) {
        VERBOSE_ONLY(std::fprintf(stderr, "center %zu has count %g\n", i, counts[i]);)
        if(counts[i]) {
#ifdef DUMBAVE
            row(centers, i BLAZE_CHECK_DEBUG) *= (1. / counts[i]);
#endif
        } else {
            if(!costs) {
                std::srand(std::time(nullptr));
                costs.reset(new typename MatrixType::ElementType[nr]);
                OMP_PFOR
                for(size_t j = 0; j < nr; ++j) {
                    //if(j == i) costs[j] = 0.; else
                    costs[j] = func(row(centers, assignments[j]), row(data, j)) * getw(j);
                    //std::fprintf(stderr, "costs[%zu] = %g\n", j, costs[j]);
                }
            }
            inclusive_scan(costs.get(), costs.get() + nr, costs.get());
            //for(unsigned i = 0; i < nr; ++i) std::fprintf(stderr, "%u:%g\t", i, costs[i]);
            //std::fputc('\n', stderr);
            size_t item = std::lower_bound(costs.get(), costs.get() + nr, costs[nr - 1] * double(std::rand()) / RAND_MAX) - costs.get();
            costs[item] = 0.;
            assignments[item] = i;
            //std::fprintf(stderr, "Reassigning center %zu to row %zu because it has lost all support\n", i, item);
            std::fprintf(stderr, "Reassigning center %zu to row %zu because it has lost all support\n", i, item);
            row(centers, i BLAZE_CHECK_DEBUG) = row(data, item);
            centers_reassigned = true;
        }
    }
    if(centers_reassigned)
        goto get_assignment_counts;
    std::fprintf(stderr, "Assigning to centers\n");
    // 2. Assign centers
    double total_loss = 0.;
    OMP_PRAGMA("omp parallel for reduction(+:total_loss)")
    for(size_t i = 0; i < nr; ++i) {
        auto dr = row(data, i BLAZE_CHECK_DEBUG);
        auto lhr = row(centers, 0 BLAZE_CHECK_DEBUG);
        auto dist = blz::serial(func(dr, lhr));
        unsigned label = 0;
        double newdist;
        for(unsigned j = 1;j < centers.rows(); ++j) {
            if((newdist = blz::serial(func(dr, row(centers, j BLAZE_CHECK_DEBUG)))) < dist) {
                //std::fprintf(stderr, "newdist: %g. olddist: %g. Replacing label %u with %u\n", newdist, dist, label, j);
                dist = newdist;
                label = j;
            }
        }
        assignments[i] = label;
        total_loss += getw(i) * dist;
    }
    std::fprintf(stderr, "total loss: %g\n", total_loss);
    if(std::isnan(total_loss)) total_loss = std::numeric_limits<decltype(total_loss)>::infinity();
    return total_loss;
}

template<typename IT, typename MatrixType, typename CMatrixType=MatrixType, typename WFT=double,
         typename Functor=blz::sqrL2Norm>
void lloyd_loop(std::vector<IT> &assignments, std::vector<WFT> &counts,
                CMatrixType &centers, MatrixType &data,
                double tolerance=0., size_t maxiter=-1,
                const Functor &func=Functor(),
                const WFT *weights=nullptr)
{
    if(tolerance < 0.) throw 1;
    size_t iternum = 0;
    double oldloss = std::numeric_limits<double>::max(), newloss;
    for(;;) {
        std::fprintf(stderr, "Starting iter %zu\n", iternum);
        newloss = lloyd_iteration(assignments, counts, centers, data, func, weights);
        double change_in_cost = std::abs(oldloss - newloss) / std::min(oldloss, newloss);
        if(iternum++ == maxiter || change_in_cost <= tolerance) {
            std::fprintf(stderr, "Change in cost from %g to %g is %g\n", oldloss, newloss, change_in_cost);
            break;
        }
        std::fprintf(stderr, "new loss at %zu: %0.30g. old loss: %0.30g\n", iternum, newloss, oldloss);
        oldloss = newloss;
    }
    std::fprintf(stderr, "Completed with final loss of %0.30g after %zu rounds\n", newloss, iternum);
}

template<typename IT, typename MatrixType, typename CMatrixType=MatrixType, typename WFT=double, typename Functor=blz::sqrL2Norm, typename RNG, typename SelContainer>
void minibatch_lloyd_iteration(std::vector<IT> &assignments, std::vector<WFT> &counts,
                               CMatrixType &centers, MatrixType &data, unsigned batchsize,
                               RNG &rng,
                               SelContainer &selection,
                               const Functor &func=Functor(),
                               const WFT *weights=nullptr)
{
    if(batchsize < assignments.size()) batchsize = assignments.size();
    const size_t np = assignments.size();
    selection.clear();
    selection.reserve(batchsize);
    schism::Schismatic<IT> div(np);
    double weight_sum = 0, dbs = batchsize;
    for(;;) {
        auto ind = div.mod(rng());
        if(std::find(selection.begin(), selection.end(), ind) == selection.end()) {
            blz::push_back(selection, ind);
            if((weight_sum += (weights ? weights[ind]: WFT(1))) >= dbs)
                break;
        }
    }
    shared::sort(selection.begin(), selection.end());
    std::unique_ptr<IT[]> asn(new IT[np]());
    OMP_PRAGMA("omp parallel")
    {
        std::unique_ptr<IT[]> labels(new IT[batchsize]);
        OMP_PRAGMA("omp for")
        for(size_t i = 0; i < batchsize; ++i) {
            const auto ind = selection[i];
            const auto dr = row(data, ind);
            const auto lhr = row(centers, 0 BLAZE_CHECK_DEBUG);
            double dist = blz::serial(func(dr, lhr)), newdist;
            IT label = 0;
            for(unsigned j = 1; j < centers.rows(); ++j)
                if((newdist = blz::serial(func(dr, row(centers, j BLAZE_CHECK_DEBUG)))) < dist)
                    dist = newdist, label = j;
            labels[i] = label;
            OMP_ATOMIC
            counts[label] += weights ? weights[ind]: WFT(1);
        }
        OMP_PRAGMA("omp for")
        for(size_t i = 0; i < batchsize; ++i) {
            const auto label = labels[i];
            const auto ind = selection[i];
            const auto dr = row(data, ind);
            const double eta = (weights ? weights[ind]: WFT(1)) / counts[label];
            auto crow = row(centers, label BLAZE_CHECK_DEBUG);
            crow = blz::serial((1. - eta) * crow + eta * dr);
        }
    }
}

template<typename IT, typename MatrixType, typename CMatrixType=MatrixType, typename WFT=double,
         typename Functor=blz::sqrL2Norm>
double mb_lloyd_loop(std::vector<IT> &assignments, std::vector<WFT> &counts,
                     CMatrixType &centers, MatrixType &data,
                     unsigned batch_size,
                     size_t maxiter=10000,
                     const Functor &func=Functor(),
                     uint64_t seed=137,
                     const WFT *weights=nullptr)
{
    std::vector<IT> selection;
    selection.reserve(batch_size);
    wy::WyRand<IT, 4> rng(seed);
    size_t iternum = 0;
    while(iternum++ < maxiter) {
        std::fprintf(stderr, "Starting minibatch iter %zu\n", iternum);
        minibatch_lloyd_iteration(assignments, counts, centers, data, batch_size, rng, selection, func, weights);
    }
    double loss = 0.;
    std::memset(counts.data(), 0, sizeof(counts[0]) * counts.size());
    // TODO: Consider moving the centers as well at this step.
    OMP_PRAGMA("omp parallel for reduction(+:loss)")
    for(size_t i = 0; i < assignments.size(); ++i) {
        auto dr = row(data, i BLAZE_CHECK_DEBUG);
        double closs = func(dr, row(centers, 0 BLAZE_CHECK_DEBUG)), newloss;
        IT label = 0;
        for(unsigned j = 1; j < centers.rows(); ++j)
            if((newloss = func(dr, row(centers, j BLAZE_CHECK_DEBUG))) < closs)
                closs = newloss, label = j;
        OMP_ATOMIC
        ++counts[label];
        assignments[i] = label;
        loss += closs;
    }
    std::fprintf(stderr, "Completed with final loss of %0.30g after %zu rounds\n", loss, iternum);
    return loss;
}



template<typename Iter,
         typename IT=std::uint32_t, typename RNG=wy::WyRand<uint32_t, 2>,
         typename FT=ContainedTypeFromIterator<Iter>, typename Distance=sqrL2Norm>
auto kmeans_coreset(Iter start, Iter end,
                    size_t k, RNG &rng,
                    size_t cs_size,
                    const FT *weights=nullptr, const Distance &dist=Distance()) {
    auto [centers, assignments, sqdists] = kmeanspp(start, end, rng, k, dist);
    using sq_t = typename decltype(sqdists)::value_type;
    coresets::CoresetSampler<sq_t, IT> cs;
    size_t np = end - start;
    cs.make_sampler(np, centers.size(), sqdists.data(), assignments.data(), weights,
                    /*seed=*/rng());
    auto ics(cs.sample(cs_size, rng()));
    DBG_ONLY(for(auto i: ics.indices_) assert(i < np);)
    static_assert(std::is_same<decltype(ics), coresets::IndexCoreset<IT, sq_t>>::value, "must be this type");
    //coresets::IndexCoreset<IT, sq_t> ics(cs.sample(cs_size, rng()));
#ifndef NDEBUG
    std::fprintf(stderr, "max sampled idx: %u\n", *std::max_element(ics.indices_.begin(), ics.indices_.end()));
#endif
    return ics;
}

template<typename FT, bool SO,
         typename IT=std::uint32_t, typename RNG=wy::WyRand<uint32_t, 2>>
auto kmeans_index_coreset(const blaze::DynamicMatrix<FT, SO> &mat, size_t k, RNG &rng, size_t cs_size,
                           const FT *weights=nullptr, bool rowwise=true)
{
    if(!rowwise) throw std::runtime_error("Not implemented");
    const auto &blzview = reinterpret_cast<const blz::DynamicMatrix<FT, SO> &>(mat);
    return kmeans_coreset(blzview.rowiterator().begin(), blzview.rowiterator().end(),
                          k, rng, cs_size, weights);
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
    for(auto idx: ics.indices_)
        assert(idx < rowwise ? mat.rows(): mat.columns());
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
