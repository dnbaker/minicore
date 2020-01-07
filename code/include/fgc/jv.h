#pragma once
#ifndef JAIN_VAZIRANI_H__
#define JAIN_VAZIRANI_H__
#include <queue>
#include <vector>
#include <iostream>
#include "graph.h"
#include "blaze_adaptor.h"
#include "jv_solver.h"

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
std::vector<typename Graph::vertex_descriptor>
    jain_vazirani_kmedian(Graph &x,
                          const std::vector<typename Graph::vertex_descriptor> &candidates,
                          unsigned k)
{
    // candidates consists of a vector of potential facility centers.
    //using Edge = typename Graph::edge_descriptor;
    const size_t n = x.num_vertices();
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
    std::cerr << "cost matrix: " << c << '\n';
    // maxcost = (maxcostedgecost * num_cities)
    
    NaiveJVSolver<float> jvs(c.rows(), c.columns(), 0.);
#if 0
    auto tmp = jvs.kmedian(c, k, true);
    std::vector<typename Graph::vertex_descriptor> ret; ret.reserve(tmp.size());
    for(const auto v: tmp) ret.push_back(v);
    return ret;
#else
    return jvs.kmedian(c, k);
#endif
    //auto oneopen = jvs.ufl(c, cost_ubound);
    //auto allopen = jvs.ufl(c, cost_lbound);
} // jain_vazirani_ufl

#if 0
template<typename Graph>
auto jain_vazirani_kmedian(Graph &x,
                           const std::vector<typename Graph::vertex_descriptor> &candidates,
                           size_t k) {
    double minval = 0.;
    auto edge_iterator_pair = x.edges();
    typename property_map<Graph, edge_weight_t>::type weightmap = get(edge_weight, x);
    auto maxelit = std::max_element(edge_iterator_pair.begin(), edge_iterator_pair.end(),
                                    [&weightmap](auto x, auto y) {return weightmap[x] < weightmap[y];});
    double maxcost = weightmap[*maxelit];
    double maxval = maxcost * x.num_vertices();
    auto ubound = jain_vazirani_ufl(x, candidates, 0); // Opens all facilities
    auto lbound = jain_vazirani_ufl(x, candidates, maxval);
    while(std::min(k - ubound.size(), lbound.size() - k) < k / 2) {
        // do binary search
        // on cost
        throw std::runtime_error("NotImplemented");
    }
} // jain_vazirani_kmedian
#endif


} // namespace fgc

#endif /* JAIN_VAZIRANI_H__ */
