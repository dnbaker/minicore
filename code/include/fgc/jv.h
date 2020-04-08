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

template<typename Graph, typename Mat>
void fill_cand_distance_mat(const Graph &x, Mat &mat, const std::vector<typename Graph::vertex_descriptor> &candidates) {
    assert(mat.rows() == candidates.size());
    assert(mat.columns() == x.num_vertices());
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0; i < candidates.size(); ++i) {
        auto edge = candidates[i];
        auto r = row(mat, i);
        assert(r.size() == x.num_vertices());
        boost::dijkstra_shortest_paths(x, edge,
                                       distance_map(&r[0]));
        if(r[0] == std::numeric_limits<float>::max()) {
            throw std::runtime_error("This is probably not connected");
        }
        // Now the row c(r, i) has the distances from candidate facility candidates[i] to
        // all nodes.
    }
}

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
                          unsigned k,
                          blaze::DynamicMatrix<float> *costsmat=nullptr)
{
    // candidates consists of a vector of potential facility centers.
    //using Edge = typename Graph::edge_descriptor;
    const size_t n = x.num_vertices();
    size_t nf = candidates.size();
    std::unique_ptr<blaze::DynamicMatrix<float>> optr;
    if(costsmat) {
        costsmat->resize(nf, n);
    } else {
        optr.reset(new blaze::DynamicMatrix<float>(nf, n, 0.));
    }
    blaze::DynamicMatrix<float> &c(*(costsmat ? costsmat: optr.get()));
    fill_cand_distance_mat(x, c, candidates);
#if VERBOSE_AF
    std::cerr << "cost matrix: " << c << '\n';
#endif
    // maxcost = (maxcostedgecost * num_cities)

    jv::JVSolver<blaze::DynamicMatrix<float>> jvs(c);
    auto [sol, solasn] = jvs.kmedian(k);
    std::vector<typename Graph::vertex_descriptor> ret(sol.size());
    for(size_t i = 0; i < sol.size(); ++i)
        ret[i] = candidates[sol[i]];
    return ret;
} // jain_vazirani_ufl

} // namespace fgc

#endif /* JAIN_VAZIRANI_H__ */
