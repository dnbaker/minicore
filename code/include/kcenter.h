#pragma once
#include "kmeans.h"
#include "coreset.h"
#include "alias_sampler/div.h"
#include "flat_hash_map/flat_hash_map.hpp"
#include <queue>

namespace clustering {
template<typename T, typename H = std::hash<T>, typename E = std::equal_to<T>, typename A = std::allocator<T> >
using flat_hash_set = ska::flat_hash_set<T, H, E, A>;
using std::partial_sum;
using blz::L2Norm;

// Replace 
template<typename Iter, typename FT=ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG, typename Norm=L2Norm>
std::vector<IT>
kcenter_greedy_2approx(Iter first, Iter end, RNG &rng, size_t k, const Norm &norm=Norm())
{
    static_assert(sizeof(typename RNG::result_type) == sizeof(IT), "IT must have the same size as the result type of the RNG");
    // Greedy 2-approximation
    static_assert(std::is_arithmetic<FT>::value, "FT must be arithmetic");
    size_t np = end - first;
    std::vector<IT> centers;
    std::vector<FT> distances(np, 0.), cdf(np);
    {
        auto fc = rng() % np;
        centers.push_back(fc);
        auto &lhs = first[centers.front()];
#ifdef _OPENMP
        OMP_PRAGMA("omp parallel for")
        for(size_t i = 0; i < np; ++i) {
            distances[i] = norm(lhs, first[i]);
        }
#else
        SK_UNROLL_8
        for(size_t i = 0; i < np; ++i) {
            distances[i] = norm(lhs, first[i]);
        }
#endif
        assert(distances[fc] == 0.);
    }
        
    while(centers.size() < k) {
        auto newc = std::max_element(distances.begin(), distances.end()) - distances.begin();
        centers.push_back(newc);
        const auto &lhs = first[newc];
        OMP_PRAGMA("omp parallel for")
        for(IT i = 0; i < np; ++i) {
            auto &ldist = distances[i];
            if(auto dist(norm(lhs, first[i])); ldist < dist) ldist = dist;
        }
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
template<typename IT=std::uint32_t, typename Container=std::vector<::std::uint32_t>,
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


/*
// Algorithm 1 from the above DYW paper
// Z = # outliers
// \mu = quality of coreset
// size of coreset: 2z + O((2/\mu)^p k)
// \gamma = z / n
*/

template<typename Iter, typename FT=ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG, typename Norm=L2Norm>
std::tuple<std::vector<IT>, std::vector<IT>, std::vector<IT>, double>
kcenter_bicriteria(Iter first, Iter end, RNG &rng, size_t k, double eps,
                   double gamma=0.001, size_t t = 100, double eta=0.01,
                   const Norm &norm=Norm())
{
    // Step 1: constants
    auto np = end - first;
    const size_t z = std::ceil(gamma * np);
    const size_t samplechunksize = std::ceil(std::log(1./eta) / (1 - gamma));
    const size_t farthestchunksize = std::ceil((1 + eps) * z);
    std::vector<IT> ret;
    std::vector<IT> labels(np);
    ret.reserve(samplechunksize);
    std::vector<FT> distances(np);
    // randomly select 'log(1/eta) / (1 - eps)' vertices from X and add them to E.
    while(ret.size() < samplechunksize) {
        // Assuming that this is relatively small and we can take bad asymptotic complexity
        auto newv = rng() % np;
        if(std::find(ret.begin(), ret.end(), newv) == ret.end())
            ret.push_back(newv);
    }
    if(samplechunksize > 100) {
        std::fprintf(stderr, "Warning: with samplechunksize %zu, it may end up taking a decent amount of time. Consider swapping this in for a hash set.", samplechunksize);
    }
    detail::fpq<IT> pq;
    pq.reserve(farthestchunksize + 1);
    // Fill the priority queue from the first set
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0; i < np; ++i) {
        const auto &ref = first[ret[i]];
        double dist = norm(ref, first[ret.first()]);
        double newdist;
        IT label = 0; // This label is an index into the ret vector, rather than the actual index
        for(size_t j = 1, e = ret.size(); j < e; ++j) {
            if((newdist = norm(ref, first[ret[j]])) < dist) {
                label = j;
                dist = newdist;
            }
        }
        distances[i] = dist;
        labels[i] = ret[label];
        if(pq.empty() || dist > pq.top().first) {
            const auto p = std::make_pair(dist, i);
            OMP_PRAGMA("omp critical")
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
    std::vector<IT> random_samples(samplechunksize);
    // modulo without a div/mod instruction, much faster
    schism::Schismatic<IT> div(farthestchunksize); // pq size
    for(size_t j = 0;;) {
        // Sample 'samplechunksize' points from pq into random_samples.
        // Sample them
        size_t rsi = 0;
        IT *rsp = random_samples.data();
        do {
            IT index = div.mod(rng());
            // (Without replacement)
            if(std::find(rsp, rsp + rsi, index) == rsp + rsi)
                rsp[rsi++] = index;
        } while(rsi < samplechunksize);
        // random_samples now contains indexes *into pq*
        std::transform(rsp, rsp + rsi, rsp, [pqi=pq.getc().data()](auto x) {return pqi[x];});
        // random_samples now contains indexes *into original dataset*

        // Insert into solution
        ret.insert(rsp, rsp + rsi);

        // compare each point against all of the new points
        pq.getc().clear(); // empty priority queue
        OMP_PRAGMA("omp parallel for")
        for(size_t i = 0; i < np; ++i) {
            const auto &ref = first[i];
            double dist = distances[i];
            double newdist;
            IT label = labels[i];
            for(size_t j = 0; j < rsi; ++j) {
                if((newdist = norm(ref, first[rsp[j]])) < dist)
                    dist = newdist, label = rsp[j];
            }
            distances[i] = dist;
            labels[i] = label;
            if(pq.empty() || dist > pq.top().first) {
                const auto p = std::make_pair(dist, i);
                OMP_PRAGMA("omp critical")
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
        // to technically only do 't' rounds
        // Now fill the next priority queue
    }
    const double minmaxdist = pq.top().first;
    return std::make_tuple(ret, labels, std::move(pq.getc()), minmaxdist); // center ids, label assignments for all points, and the distance of the closest excluded point
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
        OMP_PRAGMA("omp parallel for")
        for(size_t i = 0; i < np; ++i) {
            const auto &ref = first[i];
            double dist = distances[i];
            double newdist;
            if((newdist = norm(first[i], newel)) < dist) {
                dist = newdist;
            }
            distances[i] = dist;
            if(pq.empty() || dist > pq.top().first) {
                const auto p = std::make_pair(dist, i);
                OMP_PRAGMA("omp critical")
                {
                    if(pq.empty() || dist > pq.top().first) {
                        pq.push(p);
                        if(pq.size() > farthestchunksize) pq.pop();
                    }
                }
            }
        }

        // Sample point
        newc = pq.getc()[rng() % farthestchunksize];
        ret.push_back(newc);
        pq.getc().clear();
    } while(ret.size() < k);
    return ret;
}// kcenter_greedy_2approx_outliers
// Algorithm 3 (coreset construction)
template<typename Iter, typename FT=ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG, typename Norm=L2Norm>
Coreset<IT, FT> 
kcenter_coreset(Iter first, Iter end, RNG &rng, size_t k, double eps, double mu,
                double rho,
                double gamma=0.001, double eta=0.01, const Norm &norm=Norm()) {
    // rho is 'D' for R^D (http://www.wisdom.weizmann.ac.il/~robi/teaching/2014b-SeminarGeometryAlgorithms/lecture1.pdf)
    // in Euclidean space, as worst-case, but usually better in real data with structure.
    assert(mu > 0. && mu <= 1.);
    size_t L = std::ceil(std::pow(2. / mu, rho) * k);
    size_t nrounds = std::ceil((L + std::sqrt(L)) / (1. - eta));
    auto [centers, labels, outliers, rtilde] = kcenter_bicriteria(first, end, rng, k, eps,
                                                                  gamma, L, eta, norm);
    //std::vector<size_t> counts(centers.size());
    flat_hash_map<IT, uint32_t> counts;
    counts.reserve(centers.size());
    throw std::runtime_error("this does not account for ignoring the Z > r");
    size_t i = 0;
    for(const auto outlier: outliers) {
        while(i < outlier) {
             ++counts[labels[i++]];
        }
        ++i; // skip the outliers
    }
    while(i < np)
         ++counts[labels[i++]];
    Coreset<IT, FT> ret(centers.size() + outliers.size());
    for(i = 0; i < outliers.size(); ++i) {
        ret.indices_[i] = outliers[i];
        ret.weights_[i] = 1.;
    }
    for(const auto &pair: counts) {
        ret.weights_[i] = pair.second;
        ret.indices_[i] = pair.first;
    }
#if 0
Denote by r the maximum distance between E and X by
excluding the farthest 2z vertices, after the final round of Algorithm 1.
3. Let Xr = {p | p ∈ X and d(p,E) ≤ r ̃}.
4. For each vertex p ∈ Xr ̃, assign it to its nearest neighbor in E; for each vertex q ∈ E, let its weight be the
number of vertices assigning to it.
5. Add X \Xr ̃ to E; each vertex of X \Xr ̃ has weight 1.
Output E as the coreset.

#endif
    return ret;
}
}// namespace outliers

} // clustering
