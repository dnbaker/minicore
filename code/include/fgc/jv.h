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
    blaze::DynamicMatrix<float> c(nf, n, 0.);
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0; i < candidates.size(); ++i) {
        auto edge = candidates[i];
        auto r = row(c, i);
        assert(r.size() == n);
#if GETPREDS
        auto p(std::make_unique<typename Graph::vertex_descriptor[]>(n));
#endif
        boost::dijkstra_shortest_paths(x, edge,
                                       distance_map(&r[0])
#if GETPREDS
                                       .predecessor_map(&p[0])
#endif
        );
        if(r[0] == std::numeric_limits<float>::max()) {
#if defined(GETPREDS) && defined(VERBOSE_AF)
            for(size_t i = 0; i < n; ++i) {
                std::fprintf(stderr, "pi: %zu. candidate: %zu\n", size_t(pi[i]), size_t(edge));
            }
#endif
            throw 1;
        }
        // Now the row c(r, i) has the distances from candidate facility candidates[i] to
        // all nodes.
    }
    std::cerr << "cost matrix: " << c << '\n';
    // maxcost = (maxcostedgecost * num_cities)
    
    NaiveJVSolver<float> jvs(c.rows(), c.columns(), 0.);
    auto sol = jvs.kmedian(c, k);
    for(const auto v: sol)
        assert(v <= candidates.size());
    std::vector<typename Graph::vertex_descriptor> ret(sol.size());
    for(size_t i = 0; i < sol.size(); ++i)
        ret[i] = candidates[sol[i]];
    return ret;
} // jain_vazirani_ufl

} // namespace fgc

#endif /* JAIN_VAZIRANI_H__ */
