#pragma once
#ifndef FGC_STREAMING_H__
#define FGC_STREAMING_H__
#include <iostream>
#include <cstdio>
#include <type_traits>
#include <cmath>
#include <stdexcept>
#include <stack>
#include <random>
#include <zlib.h>

namespace fgc {
namespace streaming {


template<typename FT=double>
struct UniformW {
    FT operator()() const {return FT(1.);};
};
template<typename FT=double>
struct PointerW {
    const FT *data_;
    PointerW(const FT *data): data_(data) {}
    FT operator()() {return *data_++;}
};

template<typename FT=double>
struct ZlibW {
    gzFile fp_;
    ZlibW(gzFile fp): fp_(fp) {}
    FT operator()() {
        FT ret;
        if(gzread(fp_, &ret, sizeof(ret)) != sizeof(ret)) {
            throw std::runtime_error("Failed to read from file\n");
        }
        return ret;
    };
};

template<typename FT=double>
struct istreamW {
    std::istream &s_;
    istreamW(std::istream &s): s_(s) {}
    FT operator()() {
        FT ret;
        s_.read(&ret, sizeof(ret));
        return ret;
    };
};
template<typename CT>
struct is_uniform_weighting: public std::false_type {};

template<typename FT>
struct is_uniform_weighting<UniformW<FT>>: public std::true_type {};

/*
 *
 *  KServiceClusterer takes a stream (Generator) and processes it according to BMORST *.
 *  The gamma parameter is determined by an upper bound of the optimal cost
 *  and alpha, the relaxation of the triangle inequality.
 *  beta = 2 alpha^{2}c_{OFL} + 2 alpha
 *  gamma = max{4 alpha^3 c_{OFL}^2, beta * k_{OFL} + 1}
 *  c_{OFL} is 8 for the uniform case and 33 for the nonuniform case**
 *  Not knowing how to determine k_{OFL} (oversubscription of potential centers over k log(k) * alpha), I just picked 5.
 *
 *  *  Streaming k-means on well-clusterable data, Braverman, Meyerson, Ostrovsky, Roytman, Shindler, Tagiku
       https://web.cs.ucla.edu/~rafail/PUBLIC/116.pdf
 *  ** Online Facility Location, Adam Meyerson
       http://web.cs.ucla.edu/~awm/papers/ofl.pdf
    process(Generator, WeightGenerator) processes the full stream in Generator, such that Generator
    returns a const reference to the object to be added, and WeightGenerator returns a real-valued weight.
    
    WeightGenerator defaults to UniformW, which returns 1. each time. ZlibW, istreamW, and PointerW load
    successive values from a stream as described by their names.

    When the stream has been completely processed, the set of points on the sack (mstack_)
    must be clustered by another method.
    For general metric spaces, we recommend Jain-Vazirani or local search.
    For k-means, k-medians, and Bregman Divergences, we recommend Lloyd's algorithm/EM.
 */

template<typename Func, typename Item, typename WT=double, typename RNG=std::mt19937_64>
class KServiceClusterer {
    Func func_;
    WT l_i_, f_, cost_, beta_, gamma_;
    unsigned k_;
    size_t n_, i_ = 0;
public:
    struct mutable_stack: public std::stack<std::pair<Item, WT>> {
        mutable_stack() {}
        auto &getc() {return this->c;}
        const auto &getc() const {return this->c;}
    };
    mutable_stack mstack_;
    typename mutable_stack::container_type readingstack_;
    std::uniform_real_distribution<WT> urd_;
    RNG rng_;
    template<typename AItem>
    std::pair<unsigned, WT> assign(const AItem &item) const {
        if(mstack_.empty()) return {-1, std::numeric_limits<WT>::max()};
        unsigned i = 0;
        WT mindist = func(mstack_[0], item), dist;
        for(unsigned j = 1; j < mstack_.size(); ++j)
            if((dist = func(mstack_[j])) < mindist) mindist = dist, i = j;
        return {i, mindist};
    }
    template<typename AItem>
    void add(const AItem &item, WT weight=1.) {
        auto rv = urd_(rng_);
        auto [asn, mincost] = assign(item);
        auto cost = weight * mincost;
        if(cost / f_ > urd_(rng_)) {
            mstack_.push(std::make_pair<Item, WT>(item, weight));
        } else {
            cost_ += cost;
            mstack_[asn].second += weight;
        }
        if(cost_ > gamma_ * l_i_ || mstack_.size() > (gamma_ - 1) * (1 + std::log(n_)) * k_) {
            readingstack_ = std::move(mstack_.getc());
            ++i_;
            l_i_ *= beta_;
            f_ = l_i_ / (k_ * (1 + std::log(n_)));
        }
    }
    template<typename Generator, typename WeightGen=UniformW<WT>>
    void process_step(Generator &gen, WeightGen &wgen=WeightGen()) {
        if(readingstack_.size()) {
            add(readingstack_.top().first, readingstack_.top().second);
            readingstack_.pop();
        }
        add(gen(), wgen());
    }
    template<typename Generator, typename WeightGen=UniformW<WT>>
    void process(Generator &gen, WeightGen &&wgen=WeightGen()) {
        WeightGen weight_gen(std::move(wgen));
        while(gen.size() /* maybe rename for easier interface? */ ) {
            process_step(gen, wgen);
        }
    }
    KServiceClusterer(Func func, unsigned k, size_t n, double gamma, double beta):
        func_(func),
        beta_(beta), gamma_(gamma), l_i_(1), k_(k), n_(n), i_(1) {
        rng_.seed(std::rand());
    }
};

template<typename Func, template<typename> class WeightGen=UniformW, typename FT=double>
auto make_kservice_clusterer(Func func, unsigned k, size_t n, double alpha, WeightGen<FT> &wg) {
    double cofl = is_uniform_weighting<WeightGen<FT>>::value ? 8: 20;
    double kofl = is_uniform_weighting<WeightGen<FT>>::value ? 5: 22;
    double beta = 2 * alpha * alpha * cofl + 2 * alpha;
    double gamma = std::max(4. * std::pow(alpha, 3) * cofl * cofl, beta * kofl + 1.);
    KServiceClusterer(std::move(func), k, n, gamma, beta);
}

} // namespace streaming
} // namespace fgc

#endif /* FGC_STREAMING_H__ */
