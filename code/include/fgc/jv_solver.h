#ifndef JV_SOLVER_H__
#define JV_SOLVER_H__
#include "blaze_adaptor.h"
#include "pdqsort/pdqsort.h"
#include <stdexcept>

namespace fgc {

struct edgetup: public std::tuple<double, uint32_t, uint32_t> {
    // Consists of:
    // 1. Cost of edge
    // 2. Facility index
    // 3 Distance index.
    // Can be easily accessed with these member functions:
    template<typename...A> edgetup(A &&...args): std::tuple<double, uint32_t, uint32_t>(std::forward<A>(args)...) {}
    auto cost() const {return std::get<0>(*this);}
    auto &cost() {return std::get<0>(*this);}
    auto fi() const {return std::get<1>(*this);}
    auto &fi() {return std::get<1>(*this);}
    auto di() const {return std::get<2>(*this);}
    auto &di() {return std::get<2>(*this);}
    auto sprintf(char *buf) const {
        return std::sprintf(buf, "%f:%u:%u", cost(), fi(), di());
    }
};

template<typename FT>
struct NaiveJVSolver {
    // Complexity: something like F^2N + N^2F
    // Much slower than it should be
    blz::DM<FT> w_;
    blz::DV<FT> v_;
    blz::DV<uint32_t> numconnected_, numtight_;
    std::vector<edgetup> edges_;
    double facility_cost_, maxalph_;
    std::unordered_set<uint32_t> S_, tempopen_, nottempopen_;
    NaiveJVSolver(size_t nf, size_t nc, double fc=1.):
        w_(nf, nc, 0), v_(nc, 0), numconnected_(nf, 0), numtight_(nf, 0), edges_(nf * nc), facility_cost_(fc), maxalph_(0)
    {
    }
    void reset(double newfacility_cost) {
        if(newfacility_cost) facility_cost_ = newfacility_cost;
        reset(w_); reset(v_); reset(numconnected_);
        maxalph_ = 0;
        nottempopen_.clear();
        for(size_t i = 0; i < w_.rows(); ++i) nottempopen_.insert(i);
        tempopen_.clear();
    }
    template<typename MatType>
    void setup(const MatType &mat) {
        if(mat.rows() != w_.rows() || mat.columns() != w_.columns()) {
            char buf[256];
            std::sprintf(buf, "Wrong number of rows or columns: received %zu/%zu, expected %zu/%zu\n", mat.rows(), mat.columns(), w_.rows(), w_.columns());
            throw std::runtime_error(buf);
        }
        OMP_PRAGMA("omp parallel for")
        for(size_t i = 0; i < mat.rows(); ++i) {
            auto p = &edges_[i * mat.columns()];
            for(size_t j = 0; j < mat.columns(); ++j) {
                p[j] = {mat(i, j), i, j};
            }
        }
        pdqsort(&edges_[0], &edges_[edges_.size()], [](const auto x, const auto y) {return x.cost() > y.cost();});
        // Sorted in reverse order, so edges_.back() is the minimum cost
        // and edges_.pop_back() pops it off.
        nottempopen_.clear();
        for(size_t i = 0; i < mat.rows(); ++i) nottempopen_.insert(i);
    }
    template<typename MatType>
    std::vector<uint64_t> phase2(const MatType &mat) { // Electric Boogaloo
        wy::WyRand<uint32_t, 2> rng(mat.rows());
        std::vector<uint32_t> tov(tempopen_.begin(), tempopen_.end());
        std::vector<uint32_t> to_remove;
        auto lai = rng() % tov.size();
        auto la = tov[lai];
        std::swap(tov[lai], tov.back());
        tov.pop_back();
        std::vector<uint64_t> ret{la};
        while(tov.size()) {
            auto r = row(mat, la);
            for(size_t i = 0; i < mat.columns(); ++i) {
                if(r[i] > 0.) {
                    for(size_t j = 0; j < mat.rows(); ++j)
                        if(mat(j, i) > 0.)
                            to_remove.push_back(j);
                }
            }
            for(const auto item: to_remove)
                tempopen_.erase(item);
            to_remove.clear();
            if(tempopen_.empty()) break;
            auto ci = rng() % tov.size();
            auto cv = tov[ci];
            ret.push_back(cv);
        }
        return ret;
    }
    template<typename MatType, typename CT>
    double calculate_cost(const MatType &mat, const CT &open_facilities) const {
        if(open_facilities.empty()) return std::numeric_limits<double>::max();
        double faccost = open_facilities.size() * facility_cost_;
        double citycost = blz::sum(blz::min<blz::columnwise>(rows(mat, open_facilities, blaze::unchecked)));
        return citycost + faccost;
    }
    template<typename MatType>
    std::vector<uint64_t> ufl(const MatType &mat, double faccost) {
        // Uncapacited Facility Location problem with facility cost = faccost
        facility_cost_ = faccost;
        for(const auto edge: edges_) {
            char buf[30];
            edge.sprintf(buf);
            std::fprintf(stderr, "edge: %s|\n", buf);
            //std::fputc('\n');
        }
        phase1(mat);
        return phase2(mat);
    }
    template<typename MatType>
    std::vector<uint64_t> kmedian(const MatType &mat, unsigned k, bool perform_setup=false) {
        if(perform_setup) setup(mat);
        // Uncapacited Facility Location problem with facility cost = faccost
        double maxcost = mat.columns() * max(mat);
        if(std::isinf(maxcost)) {
            maxcost = std::numeric_limits<double>::min();
            for(const auto r: blz::rowiterator(mat)) {
                for(const auto v: r)
                    if(!std::isinf(v) && v > maxcost)
                        maxcost = v;
            }
        }
        double mincost = 0.;
        double medcost = maxcost / 2;
        //auto ubound = ufl(mat, maxcost);
        //auto lbound = ufl(mat, mincost);
        auto med = ufl(mat, medcost);
        while(med.size() != k) {
            if(med.size() == k) break;
            if(med.size() > k)
                mincost = medcost; // med has too many, increase cost.
            else
                maxcost = medcost; // med has too few, lower cost.
            medcost = (mincost + maxcost) / 2.;
            med = ufl(mat, medcost);
            std::fprintf(stderr, "Solution cost: %f\n", calculate_cost(mat, med));
        }
        return med;
    }
    
    std::pair<uint32_t, double> min_tightening_cost() const {
        if(edges_.empty()) return std::make_pair(uint32_t(-1), std::numeric_limits<double>::max());
        auto edge = edges_.back();
        return std::make_pair(edge.di(), edge.cost() - maxalph_);
    }
    std::pair<uint32_t, double> min_opening_cost() const {
        double mincost = std::numeric_limits<double>::max();
        uint32_t ind = -1u;
        for(const auto fid: nottempopen_) {
            auto nsupport = std::accumulate(row(w_, fid).begin(), row(w_, fid).end(), size_t(0), [](auto x, auto y) {return x + y >= 0.;});
            if(nsupport == 0) return std::make_pair(-1u, std::numeric_limits<double>::max());
            auto availsum = std::accumulate(row(w_, fid).begin(), row(w_, fid).end(), 0.);
            std::fprintf(stderr, "rowsum: %f. facility cost: %f\n", availsum, facility_cost_);
            auto diff = facility_cost_ - availsum;
            auto cost = nsupport ? diff / nsupport: std::numeric_limits<double>::max();
            std::fprintf(stderr, "diff: %g. cost: %g\n", diff, cost);
            if(cost < mincost) mincost = cost, ind = fid;
        }
        return std::make_pair(ind, mincost);
    }
    template<typename MatType>
    void perform_increment(double inc,  std::vector<uint32_t> &to_remove, const MatType &mat) {
        maxalph_ += inc;
        for(const auto item: S_) {
            v_[item] = maxalph_;
            for(size_t fi = 0; fi < w_.rows(); ++fi) {
                if(maxalph_ >= mat(fi,item)) {
                    if(tempopen_.find(fi) != tempopen_.end()) // && std::find(to_remove.begin(), to_remove.end(), item) != to_remove.end())
                        to_remove.push_back(item);
                }
                w_(fi, item) = std::max(0., maxalph_ - mat(fi, item));
            }
        }
    }
    template<typename MatType>
    void phase1(const MatType &mat) {
        auto &S(S_);
        S.clear(); S.reserve(v_.size());
        for(size_t i = 0; i < v_.size(); S.insert(i++));
        std::vector<uint32_t> to_remove;
        std::fprintf(stderr, "Filled S\n");
        size_t nz = 0;
        while(edges_.back().cost() == 0.) {
            ++nz;
            edges_.pop_back();
        }
        std::fprintf(stderr, "nz: %zu\n", nz);
        while(S.size()) {
            std::fprintf(stderr, "Size of S: %zu\n", S.size());
            //std::fprintf(stderr, "getting min tight cost\n");
            auto [bestedge, tightinc] = min_tightening_cost();
            //std::fprintf(stderr, "got min tight cost\n");
            auto [bestfac, openinc]   = min_opening_cost();
            //std::fprintf(stderr, "got min opening cost\n");
            bool tighten = true;
            if(tightinc < openinc) edges_.pop_back();
            else tighten = false;
            const double inc = std::min(tightinc, openinc);
            std::fprintf(stderr, "inc: %g. open: %g. tighten: %g\n", inc, openinc, tightinc);
            perform_increment(inc, to_remove, mat);
            std::fprintf(stderr, "new alpha: %g\n", maxalph_);
            if(tighten) {
                while(edges_.back().cost() == maxalph_)
                    edges_.pop_back();
            } else {
                //auto fc = facility_cost_;
                tempopen_.insert(bestfac);
                nottempopen_.erase(bestfac);
                for(const auto item: S) {
                    if(v_.at(item) >= mat(bestfac, item)) // && std::find(to_remove.begin(), to_remove.end(), s) != to_remove.end())
                        to_remove.push_back(item);
                }
            } 
            for(const auto item: to_remove)
                S.erase(item);
        }
    }
};

} // namespace fgc

#endif
