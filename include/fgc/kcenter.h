#pragma once
#include "kmeans.h"
#include "matrix_coreset.h"
#include "alias_sampler/div.h"
#include <queue>

namespace fgc {
namespace coresets {
using std::partial_sum;
using blz::L2Norm;
using blz::push_back;


/*
 *
 * Greedy, provable 2-approximate solution
 * T. F. Gonzalez. Clustering to minimize the maximum intercluster distance. Theoretical Computer Science, 38:293-306, 1985.
 */
template<typename Iter, typename FT=ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG, typename Norm=L2Norm>
std::vector<IT>
kcenter_greedy_2approx(Iter first, Iter end, RNG &rng, size_t k, const Norm &norm=Norm(), size_t maxdest=0)
{
    static_assert(sizeof(typename RNG::result_type) == sizeof(IT), "IT must have the same size as the result type of the RNG");
    static_assert(std::is_arithmetic<FT>::value, "FT must be arithmetic");
    auto dm = make_index_dm(first, norm);
    size_t np = end - first;
    if(maxdest == 0) maxdest = np;
    std::vector<IT> centers(k);
    std::vector<FT> distances(np, 0.);
    VERBOSE_ONLY(std::fprintf(stderr, "[%s] Starting kcenter_greedy_2approx\n", __PRETTY_FUNCTION__);)
    {
        auto fc = rng() % maxdest;
        centers[0] = fc;
        distances[fc] = 0.;
        OMP_ELSE(OMP_PFOR, SK_UNROLL_8)
        for(size_t i = 0; i < maxdest; ++i) {
            if(unlikely(i == fc)) continue;
            distances[i] = dm(fc, i);
        }
        assert(distances[fc] == 0.);
    }

    for(size_t ci = 1; ci < k; ++ci) {
        auto it = std::max_element(distances.begin(), distances.end());
        VERBOSE_ONLY(std::fprintf(stderr, "maxelement is %zd from start\n", std::distance(distances.begin(), it));)
        uint64_t newc = it - distances.begin();
        centers[ci] = newc;
        distances[newc] = 0.;
        OMP_PFOR
        for(IT i = 0; i < maxdest; ++i) {
            if(unlikely(i == newc)) continue;
            auto &ldist = distances[i];
            const auto dist = dm(newc, i);
            if(dist < ldist) {
                ldist = dist;
            }
        }
        assert(std::find_if(distances.begin(), distances.end(), [](auto x) {return std::isnan(x) || std::isinf(x);})
               == distances.end());
    }
    return centers;
} // kcenter_greedy_2approx

namespace outliers {

/*
// All algorithms in this namespace are from:
// Greedy Strategy Works for k-Center Clustering with Outliers and Coreset Construction
// Hu Ding, Haikuo Yu, Zixiu Wang
*/

namespace detail {
template<typename IT=std::uint32_t, typename Container=std::vector<std::pair<double, IT>>,
         typename Cmp=std::greater<>>
struct fpq: public std::priority_queue<std::pair<double, IT>, Container, Cmp> {
    // priority queue providing access to underlying constainer with getc()
    // , a reserve function and that defaults to std::greater<> for farthest points.
    using super = std::priority_queue<std::pair<double, IT>, Container, Cmp>;
    template<typename...Args>
    fpq(Args &&...args): super(std::forward<Args>(args)...) {}
    void reserve(size_t n) {this->c.reserve(n);}
    auto &getc() {return this->c;}
    const auto &getc() const {return this->c;}
};
} // detail



template<typename IT>
struct bicriteria_result_t: public std::tuple<IVec<IT>, IVec<IT>, std::vector<std::pair<double, IT>>, double> {
    using super = std::tuple<IVec<IT>, IVec<IT>, std::vector<std::pair<double, IT>>, double>;
    template<typename...Args>
    bicriteria_result_t(Args &&...args): super(std::forward<Args>(args)...) {}
    auto &centers() {return std::get<0>(*this);}
    auto &assignments() {return std::get<1>(*this);}
    // alias
    auto &labels() {return assignments();}
    auto &outliers() {return std::get<2>(*this);}
    double outlier_threshold() const {return std::get<3>(*this);}
    size_t num_centers() const {return centers().size();}
};

/*
// Algorithm 1 from the above DYW paper
// Z = # outliers
// \mu = quality of coreset
// size of coreset: 2z + O((2/\mu)^p k)
// \gamma = z / n
*/

template<typename Iter, typename FT=ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG, typename Norm=sqrL2Norm>
bicriteria_result_t<IT>
kcenter_bicriteria(Iter first, Iter end, RNG &rng, size_t k, double eps,
                   double gamma=0.001, size_t t = 100, double eta=0.01,
                   const Norm &norm=Norm())
{
    std::fprintf(stderr, "Note: the value k (%zu) is not used in this function or the algorithm\n", k);
    auto dm = make_index_dm(first, norm);
    // Step 1: constants
    assert(end > first);
    size_t np = end - first;
    const size_t z = std::ceil(gamma * np);
    std::fprintf(stderr, "z: %zu\n", z);
    size_t farthestchunksize = std::ceil((1 + eps) * z),
           samplechunksize = std::ceil(std::log(1./eta) / (1 - gamma));
    IVec<IT> ret;
    IVec<IT> labels(np);
    ret.reserve(samplechunksize);
    std::vector<FT> distances(np);
    // randomly select 'log(1/eta) / (1 - eps)' vertices from X and add them to E.
    while(ret.size() < samplechunksize) {
        // Assuming that this is relatively small and we can take bad asymptotic complexity
        auto newv = rng() % np;
        if(std::find(ret.begin(), ret.end(), newv) == ret.end())
            push_back(ret, newv);
    }
    assert(flat_hash_set<IT>(ret.begin(), ret.end()).size() == ret.size());
    if(samplechunksize > 100) {
        std::fprintf(stderr, "Warning: with samplechunksize %zu, it may end up taking a decent amount of time. Consider swapping this in for a hash set.", samplechunksize);
    }
    if(samplechunksize > farthestchunksize) {
        std::fprintf(stderr, "samplecc is %zu (> fcs %zu). changing gcs to scc + z (%zu)\n", samplechunksize, farthestchunksize, samplechunksize + z);
        farthestchunksize = samplechunksize + z;
    }
    detail::fpq<IT> pq;
    pq.reserve(farthestchunksize + 1);
    const auto fv = ret[0];
    labels[fv] = fv;
    distances[fv] = 0.;
    // Fill the priority queue from the first set
    OMP_PFOR
    for(size_t i = 0; i < np; ++i) {
        double dist = dm(fv, i);
        double newdist;
        IT label = 0; // This label is an index into the ret vector, rather than the actual index
        for(size_t j = 1, e = ret.size(); j < e; ++j) {
            if((newdist = dm(i, ret[j])) < dist) {
                label = j;
                dist = newdist;
            }
        }
        distances[i] = dist;
        labels[i] = ret[label];
        if(pq.empty() || dist > pq.top().first) {
            const auto p = std::make_pair(dist, i);
            OMP_CRITICAL
            {
                // Check again after getting the lock
                if(pq.empty()  || dist > pq.top().first) {
                    pq.push(p);
                    if(pq.size() > farthestchunksize)
                        pq.pop();
                }
            }
        }
    }
    IVec<IT> random_samples(samplechunksize);
    // modulo without a div/mod instruction, much faster
    schism::Schismatic<IT> div(farthestchunksize); // pq size
    assert(samplechunksize >= 1.);
    for(size_t j = 0;j < t;++j) {
        //std::fprintf(stderr, "j: %zu/%zu\n", j, t);
        // Sample 'samplechunksize' points from pq into random_samples.
        // Sample them
        size_t rsi = 0;
        IT *rsp = random_samples.data();
        do {
            IT index = div.mod(rng());
            // (Without replacement)
            if(std::find(rsp, rsp + rsi, index))
                rsp[rsi++] = index;
        } while(rsi < samplechunksize);
        // random_samples now contains indexes *into pq*
        assert(pq.getc().data());
        std::transform(rsp, rsp + rsi, rsp,
            [pqi=pq.getc().data()](auto x) {
            return pqi[x].second;
        });
        for(size_t i = 0; i < rsi; ++i)
            assert(rsp[i] < np);
        // random_samples now contains indexes *into original dataset*

        // Insert into solution
#if 0
        ret.insert(ret.end(), rsp, rsp + rsi);
#else
        for(auto it = rsp, e = rsp + rsi; it < e;++it) {
            if(std::find(ret.begin(), ret.end(), *it) != ret.end()) continue;
            distances[*it] = 0.;
            labels[*it] = *it;
            ret.pushBack(*it);
        }
#endif

        // compare each point against all of the new points
        pq.getc().clear(); // empty priority queue
        // Fill priority queue
        OMP_PFOR
        for(size_t i = 0; i < np; ++i) {
            double dist = distances[i];
            if(dist == 0.) continue;
            double newdist;
            IT label = labels[i];
            for(size_t j = 0; j < rsi; ++j) {
                if((newdist = dm(i, rsp[j])) < dist)
                    dist = newdist, label = rsp[j];
            }
            distances[i] = dist;
            labels[i] = label;
            if(pq.empty() || dist > pq.top().first) {
                const auto p = std::make_pair(dist, i);
                OMP_CRITICAL
                {
                    // Check again after getting the lock in case it's changed
                    if(pq.empty() || dist > pq.top().first) {
                        pq.push(p);
                        if(pq.size() > farthestchunksize)
                        // TODO: avoid filling it all the way by checking size but it's probably not worth it
                            pq.pop();
                    }
                }
            }
        }
    }
    const double minmaxdist = pq.top().first;
    bicriteria_result_t<IT> bicret;
    assert(flat_hash_set<IT>(ret.begin(), ret.end()).size() == ret.size());
    bicret.centers() = std::move(ret);
    bicret.labels() = std::move(labels);
    bicret.outliers() = std::move(pq.getc());
    std::fprintf(stderr, "outliers size: %zu\n", bicret.outliers().size());
    std::get<3>(bicret) = minmaxdist;
    return bicret;
    // center ids, label assignments for all points besides outliers, outliers, and the distance of the closest excluded point
} // kcenter_bicriteria

/*
// Algorithm 2 from the above DYW paper
// Z = # outliers
// \gamma = z / n
*/

template<typename Iter, typename FT=ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG, typename Norm=L2Norm>
std::vector<IT>
kcenter_greedy_2approx_outliers(Iter first, Iter end, RNG &rng, size_t k, double eps,
                                double gamma=0.001,
                                const Norm &norm=Norm())
{
    auto dm = make_index_dm(first, norm);
    const size_t np = end - first;
    const size_t z = std::ceil(gamma * np);
    size_t farthestchunksize = std::ceil((1. + eps) * z);
    detail::fpq<IT> pq;
    pq.reserve(farthestchunksize + 1);
    std::vector<IT> ret;
    std::vector<FT> distances(np, std::numeric_limits<FT>::max());
    ret.reserve(k);
    auto newc = rng() % np;
    ret.push_back(newc);
    do {
        const auto &newel = first[newc];
        // Fill pq
        OMP_PFOR
        for(size_t i = 0; i < np; ++i) {
            double dist = distances[i];
            if(dist == 0.) continue;
            double newdist;
            if((newdist = dm(i, newc)) < dist)
                dist = newdist;
            distances[i] = dist;
            if(pq.empty() || dist > pq.top().first) {
                const auto p = std::make_pair(dist, i);
                OMP_CRITICAL
                {
                    if(pq.empty() || dist > pq.top().first) {
                        pq.push(p);
                        if(pq.size() > farthestchunksize) pq.pop();
                    }
                }
            }
        }

        // Sample point
        newc = pq.getc()[rng() % farthestchunksize].second;
        assert(newc < np);
        ret.push_back(newc);
        pq.getc().clear();
    } while(ret.size() < k);
    return ret;
}// kcenter_greedy_2approx_outliers
// Algorithm 3 (coreset construction)
template<typename Iter, typename FT=ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG, typename Norm=L2Norm>
coresets::IndexCoreset<IT, FT>
kcenter_coreset(Iter first, Iter end, RNG &rng, size_t k, double eps=0.1, double mu=.5,
                double rho=1.5,
                double gamma=0.001, double eta=0.01, const Norm &norm=Norm()) {
    // rho is 'D' for R^D (http://www.wisdom.weizmann.ac.il/~robi/teaching/2014b-SeminarGeometryAlgorithms/lecture1.pdf)
    // in Euclidean space, as worst-case, but usually better in real data with structure.
    assert(mu > 0. && mu <= 1.);
    const size_t np = end - first;
    size_t L = std::ceil(std::pow(2. / mu, rho) * k);
    size_t nrounds = std::ceil((L + std::sqrt(L)) / (1. - eta));
    auto bic = kcenter_bicriteria(first, end, rng, k, eps,
                                  gamma, nrounds, eta, norm);
    double rtilde = bic.outlier_threshold();
    std::fprintf(stderr, "outlier threshold: %f\n", rtilde);
    auto &centers = bic.centers();
    auto &labels = bic.labels();
    auto &outliers = bic.outliers();
#ifndef NDEBUG
    for(const auto c: centers)
        assert(c < np);
    for(const auto label: labels)
        assert(labels[label] == label);
#endif
    //std::vector<size_t> counts(centers.size());
    coresets::flat_hash_map<IT, uint32_t> counts;
    counts.reserve(centers.size());
    size_t i = 0;
    SK_UNROLL_8
    do ++counts[labels[i++]]; while(i < np);
    coresets::IndexCoreset<IT, FT> ret(centers.size() + outliers.size());
    std::fprintf(stderr, "ret size: %zu. centers size: %zu. counts size %zu. outliers size: %zu\n", ret.size(), centers.size(), counts.size(), outliers.size());
    for(i = 0; i < outliers.size(); ++i) {
        assert(outliers[i].second < np);
        ret.indices_[i] = outliers[i].second;
        ret.weights_[i] = 1.;
    }
    for(const auto &pair: counts) {
        assert(pair.first < np);
        ret.weights_[i] = pair.second;
        ret.indices_[i] = pair.first;
        ++i;
    }
    assert(i == ret.size());
    for(size_t i = 0; i < ret.indices_.size(); ++i) {
        assert(ret.indices_[i] < np);
    }
    return ret;
}
}// namespace outliers

} // coresets
} // fgc
