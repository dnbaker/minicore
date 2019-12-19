#pragma once
#ifndef JAIN_VAZIRANI_H__
#define JAIN_VAZIRANI_H__
#include <queue>
#include <vector>
#include "graph.h"
#include "blaze_adaptor.h"

/*
 * Implementation of method of Jain-Vazirani for Metric k-median clustering
 * on graphs.
 *
 *
 bibtex:
@article{Jain:2001:AAM:375827.375845,
 author = {Jain, Kamal and Vazirani, Vijay V.},
 title = {Approximation Algorithms for Metric Facility Location and k-Median Problems Using the Primal-dual Schema and Lagrangian Relaxation},
 journal = {J. ACM},
 issue_date = {March 2001},
 volume = {48},
 number = {2},
 month = mar,
 year = {2001},
 issn = {0004-5411},
 pages = {274--296},
 numpages = {23},
 url = {http://doi.acm.org/10.1145/375827.375845},
 doi = {10.1145/375827.375845},
 acmid = {375845},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {k-median problem, Lagrangian relaxation, approximation algorithms, facility location problem, linear prfgcramming},
}
 */

namespace fgc {

struct FacilityInfo {
    size_t ncities_ = 0; // number of cities contributing
    size_t t_ = std::numeric_limits<size_t>::max();       // expected time
    size_t id_; //
    INLINE bool operator<(FacilityInfo o) const {
        return t_ < o.t_;
    }
};
struct FacPQ: std::priority_queue<FacilityInfo, std::vector<FacilityInfo>> {
    using super = std::priority_queue<FacilityInfo, std::vector<FacilityInfo>>;
    template<typename...Args>
    FacPQ(Args &&...args): super(std::forward<Args>(args)...) {}
    auto       &getc()       {return this->c;}
    const auto &getc() const {return this->c;}
};

/*
 *
 * jain_vazirani_ufl is the uncapacitated facility location proble
 * from JV, where a total_cost is the budget allocated.
 * This forms the inner loop of a binary search
 * for the k-median problem.
 * The k-median solution (below) is the final step
 * in the Mikkel Thorup 12-approximate solution
 * to the metric k-median problem.
 * Implemented with additional commentary from https://www.cs.cmu.edu/~anupamg/adv-approx/lecture5.pdf
 */
template<typename Graph>
auto jain_vazirani_ufl(Graph &x,
                       const std::vector<typename Graph::vertex_descriptor> &candidates,
                       double total_cost)
{
    // candidates consists of a vector of potential facility centers.
    using Edge = typename Graph::edge_descriptor;
    const size_t n = x.num_vertices(), m = x.num_edges();
    size_t nf = candidates.size();
    blaze::DynamicMatrix<float> c(nf, n);
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0; i < candidates.size(); ++i) {
        auto edge = candidates[i];
        auto r = row(c, i);
        std::vector<typename Graph::vertex_descriptor> p(n);
        boost::dijkstra_shortest_paths(x, edge,
                                       distance_map(&r[0]).predecessor_map(&p[0]));
        // Now the row c(r, i) has the distances from candidate facility candidates[i] to
        // all nodes.
    }
    // Sort edges by weight [JV 2.4]
    std::vector<Edge> edges(x.edges().begin(), x.edges().end());
    typename property_map<Graph, edge_weight_t>::type weightmap = get(edge_weight, x);
    std::sort(edges.begin(), edges.end(), [&weightmap](auto lhs, auto rhs) {
        return weightmap[lhs] < weightmap[rhs];
    });
    // We can think of αj as the amount of money client j i willing to contribute to the solution
    // and βij as clietn j's contribution towards opening facility . From lecture5 ^ above.
    std::vector<float> alphas(n);
    blaze::DynamicMatrix<float> betas(nf, n);
    // Place them in heap... somehow update?
    FacPQ pq;
    for(size_t i = 0; i < candidates.size(); ++i)
        pq.push(FacilityInfo{0, std::numeric_limits<size_t>::max(), i});
} // jain_vazirani_ufl

auto jain_vazirani_kmedian(Graph &x,
                           const std::vector<typename Graph::vertex_descriptor> &candidates,
                           size_t k) {
    double minval = 0.;
    auto edge_iterator_pair = x.edges();
    typename property_map<Graph, edge_weight_t>::type weightmap = get(edge_weight, x);
    double maxval = weightmap[*std::max_element(edge_iterator_pair.fist, edge_iterator_pair.second,
                                      [](auto x) {return weightmap[x];})] * x.num_vertices();
    auto ubound = jain_vazirani_ufl(x, candidates, 0); // Opens all facilities
    auto lbound = jain_vazirani_ufl(x, candidates, maxval);
    while(std::min(k - ubound.size(), lbound.size() - k) < k / 2) {
        // do binary search
        // on cost
        throw std::runtime_error("NotImplemented");
    }
} // jain_vazirani_kmedian


} // namespace fgc

#endif /* JAIN_VAZIRANI_H__ */
