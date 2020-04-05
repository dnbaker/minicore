#ifndef JV_SOLVER_H__
#define JV_SOLVER_H__
#include "blaze_adaptor.h"
#include "pdqsort/pdqsort.h"
#include <stdexcept>
#include <chrono>

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
    auto print(std::FILE *ofp=stderr) {
        return std::fprintf(ofp, "%f:%u:%u", cost(), fi(), di());
    }
};

/*
 * Note:
 * The cost of opening the next facility is equal to
 * (facility_cost_ - [sum contributed from edges removed from S] ) / ([# tight edges] - [# tight edges removed from S])
 */

template<typename FT>
struct NaiveJVSolver {
    // Complexity: something like F^2N + N^2F
    // Much slower than it should be
    using DefIT = unsigned int;
    blz::DM<FT> w_;
    blz::DV<FT> v_;
    blz::DV<uint32_t> numconnected_, numtight_;
    std::vector<edgetup> edges_;
    size_t edgeindex_ = 0;
    double facility_cost_, maxalph_;
    std::unordered_set<uint32_t> S_, tempopen_, nottempopen_;
    NaiveJVSolver(size_t nf, size_t nc, double fc=1.):
        w_(nf, nc, 0), v_(nc, 0), numconnected_(nf, 0), numtight_(nf, 0), edges_(nf * nc), facility_cost_(fc), maxalph_(0)
    {
    }
    void reset(double newfacility_cost) {
        if(newfacility_cost) facility_cost_ = newfacility_cost;
        w_ = FT(0);
        //((blaze::DynamicMatrix<FT> &)w_) = FT(0);
        v_ = FT(0);
        numconnected_ = 0;
        numtight_ = 0;
        maxalph_ = 0;
        nottempopen_.clear();
        for(size_t i = 0; i < w_.rows(); ++i) nottempopen_.insert(i);
        tempopen_.clear();
    }
    template<typename MatType>
    void setup(const MatType &mat) {
        auto start = std::chrono::high_resolution_clock::now();
        if(mat.rows() != w_.rows() || mat.columns() != w_.columns()) {
            char buf[256];
            std::sprintf(buf, "Wrong number of rows or columns: received %zu/%zu, expected %zu/%zu\n", mat.rows(), mat.columns(), w_.rows(), w_.columns());
            throw std::runtime_error(buf);
        }
        OMP_PFOR
        for(size_t i = 0; i < mat.rows(); ++i) {
            auto p = &edges_[i * mat.columns()];
            auto r = row(mat, i);
            for(size_t j = 0; j < mat.columns(); ++j) {
                *p++ = {r[j], i, j};
            }
        }
        pdqsort(&edges_[0], &edges_[edges_.size()], [](const auto x, const auto y) {return x.cost() < y.cost();});
        tempopen_.clear();
        for(size_t i = 0; i < mat.rows(); ++i) nottempopen_.insert(i);
        auto stop = std::chrono::high_resolution_clock::now();
        std::fprintf(stderr, "Setup took %g\n", 0.000001 * (stop - start).count());
    }
    template<typename MatType, typename IType=DefIT>
    std::vector<IType> phase2() { // Electric Boogaloo
        std::fprintf(stderr, "tos: ntos: %zu/%zu\n", tempopen_.size(), nottempopen_.size());
        double sum = blaze::sum(w_);
        uint64_t seed;
        std::memcpy(&seed, &sum, sizeof(seed));
        wy::WyRand<uint32_t, 2> rng(seed);
        std::vector<uint32_t> tov(tempopen_.begin(), tempopen_.end());
        std::vector<uint32_t> to_remove;
        auto lai = rng() % tov.size();
        std::swap(tov[lai], tov.back());
        auto la = tov.back();
        tov.pop_back();
        std::vector<IType> ret{la};
        while(tov.size()) {
            auto r = row(w_, la);
            for(size_t i = 0; i < w_.columns(); ++i) {
                if(r[i] > 0.) {
                    for(size_t j = 0; j < w_.rows(); ++j)
                        if(w_(j, i) > 0.)
                            to_remove.push_back(j);
                }
            }
            for(const auto item: to_remove)
                tempopen_.erase(item);
            tov.assign(tempopen_.begin(), tempopen_.end());
            if(tempopen_.empty()) break;
            to_remove.clear();
            auto ci = rng() % tov.size();
            la = tov[ci];
            std::swap(tov[ci], tov.back());
            tov.pop_back();
            ret.push_back(la);
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
    template<typename MatType, typename IType=DefIT>
    std::vector<IType> ufl(const MatType &mat, double faccost) {
        // Uncapacited Facility Location problem with facility cost = faccost
        this->reset(faccost);
        assert(nottempopen_.size() == w_.rows());
        assert(tempopen_.size() == 0);
        std::fprintf(stderr, "##Starting phase1 with faccost %.12g\n", faccost);
        phase1(mat);
        return phase2<MatType, IType>();
    }
    template<typename MatType, typename IType=DefIT>
    std::vector<IType> kmedian(const MatType &mat, unsigned k, unsigned maxrounds=500) {
        setup(mat);
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
        std::fprintf(stderr, "##first solution: %zu (want k %u)\n", med.size(), k);
        size_t roundnum = 0;
        while(med.size() != k) {
            std::fprintf(stderr, "##round %zu. current size: %zu\n", ++roundnum, med.size());
            if(med.size() == k) break;
            if(med.size() > k)
                mincost = medcost; // med has too many, increase cost.
            else
                maxcost = medcost; // med has too few, lower cost.
            medcost = (mincost + maxcost) / 2.;
            auto start = std::chrono::high_resolution_clock::now();
            med = ufl<MatType, IType>(mat, medcost);
            auto stop = std::chrono::high_resolution_clock::now();
            std::fprintf(stderr, "Solution cost: %f. size: %zu. Time in ms: %g. Dimensions: %zu/%zu\n", calculate_cost(mat, med), med.size(), (stop - start).count() * 0.000001, w_.rows(), w_.columns());
            if(roundnum > maxrounds) {
                break;
            }
        }
        return med;
    }
    std::pair<uint32_t, double> min_tightening_cost() const {
        if(edgeindex_ == edges_.size()) return std::make_pair(uint32_t(-1), std::numeric_limits<double>::max());
        auto edge = edges_[edgeindex_];
        return std::make_pair(edge.di(), edge.cost() - maxalph_);
    }
    std::pair<uint32_t, double> min_opening_cost() const {
        double mincost = std::numeric_limits<double>::max();
        uint32_t ind = -1u;
        for(const auto fid: nottempopen_) {
            auto nsupport = std::accumulate(row(w_, fid).begin(), row(w_, fid).end(), size_t(0), [](auto x, auto y) {return x + y >= 0.;});
            if(nsupport == 0) return std::make_pair(-1u, std::numeric_limits<double>::max());
            auto availsum = std::accumulate(row(w_, fid).begin(), row(w_, fid).end(), 0.);
            //std::fprintf(stderr, "rowsum: %f. facility cost: %f\n", availsum, facility_cost_);
            auto diff = facility_cost_ - availsum;
            auto cost = nsupport ? diff / nsupport: std::numeric_limits<double>::max();
            //std::fprintf(stderr, "diff: %g. cost: %g\n", diff, cost);
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
        assert(nottempopen_.size() == w_.rows());
        assert(w_.rows());
        std::vector<uint32_t> to_remove;
        size_t nz = 0;
        edgeindex_ = std::find_if(edges_.begin(), edges_.end(), [](auto x) {return x.cost() > 0.;})
                     - edges_.begin();
        // Skip over 0 indices
        std::fprintf(stderr, "nz: %zu\n", nz);
        while(S.size()) {
            //std::fprintf(stderr, "Size of S: %zu. nto size: %zu. tos: %zu\n", S.size(), nottempopen_.size(), tempopen_.size());
            //std::fprintf(stderr, "getting min tight cost\n");
            auto [bestedge, tightinc] = min_tightening_cost();
            //std::fprintf(stderr, "got min tight cost\n");
            auto [bestfac, openinc]   = min_opening_cost();
            //std::fprintf(stderr, "got min opening cost\n");
            bool tighten = true;
            if(tightinc < openinc) {
                auto bec = edges_[edgeindex_].cost();
                do ++edgeindex_; while(edges_[edgeindex_].cost() == bec);
                // Skip over identical weights
            } else tighten = false;
            const double inc = std::min(tightinc, openinc);
            //std::fprintf(stderr, "inc: %g. open: %g. tighten: %g\n", inc, openinc, tightinc);
            perform_increment(inc, to_remove, mat);
            //std::fprintf(stderr, "new alpha: %g\n", maxalph_);
            if(!tighten) {
                //auto fc = facility_cost_;
                tempopen_.insert(bestfac);
                nottempopen_.erase(bestfac);
                //std::fprintf(stderr, "Inserting bestfac %u. nto size %zu. tempopen size: %zu\n", bestfac, nottempopen_.size(), tempopen_.size());
                for(const auto item: S) {
                    assert(v_.size() && v_.size() > item);
                    if(v_.at(item) >= mat(bestfac, item)) // && std::find(to_remove.begin(), to_remove.end(), s) != to_remove.end())
                        to_remove.push_back(item);
                }
            }
            for(const auto item: to_remove) {
                S.erase(item);
            }
        }
    }
};

struct Correct {
    //Correct = 1; // Fail compilation so that the branch fails until it is ready.
};
#if 0
TODO Adapt method from https://raw.githubusercontent.com/nathan-cordner/facility-location/master/fl-cpp/facility_location.C
#endif

} // namespace fgc

#endif
