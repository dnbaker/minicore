#pragma once
#ifndef FGC_STREAMING_H__
#define FGC_STREAMING_H__
#include <cstdio>
#include <cmath>

#include <iostream>
#include <type_traits>
#include <stdexcept>
#include <random>
#include <stack>


#include <zlib.h>

#include "minocore/distance.h"


namespace minocore {
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
    process(Generator, WeightGen) processes the full stream in Generator, such that Generator
    returns a const reference to the object to be added, and WeightGen returns a real-valued weight.

    WeightGen defaults to UniformW, which returns 1. each time. ZlibW, istreamW, and PointerW load
    successive values from a stream as described by their names.

    When the stream has been completely processed, the set of points on the sack (mstack_)
    must be clustered by another method.
    For general metric spaces, we recommend Jain-Vazirani or local search.
    For k-means, k-medians, and Bregman Divergences, we recommend Lloyd's algorithm/EM.
 */

template<typename Item, typename Func, typename WT=double, typename RNG=std::mt19937_64>
class KServiceClusterer {
    Func func_;
    WT l_i_, f_, cost_, alpha_, beta_;
    unsigned k_;
    size_t n_ = 0, i_ = 0;
public:
    struct mutable_stack: public std::stack<std::pair<Item, WT>> {
        mutable_stack() {}
        auto &getc() {return this->c;}
        const auto &getc() const {return this->c;}
        auto  operator[](size_t index) const {return this->c[index];}
        auto &operator[](size_t index)       {return this->c[index];}
    };
    mutable_stack mstack_;
    typename mutable_stack::container_type readingstack_;
    std::uniform_real_distribution<WT> urd_;
    RNG rng_;

    double get_cofl() const {
        return 3 * alpha_ + 1;
    }
    double get_kofl() const {
        return (6 * alpha_  + 1) * k_;
    }
    double get_gamma() const {
        return std::max(beta_ * get_kofl() + 1.,
                        4. * alpha_ * alpha_ * alpha_ * get_cofl() * get_cofl() + 2 * alpha_ * alpha_ * get_cofl());
    }

    template<typename AItem>
    std::pair<unsigned, WT> assign(const AItem &item) const {
        if(mstack_.empty()) return {-1, std::numeric_limits<WT>::max()};
        unsigned i = 0;
        WT mindist = func_(mstack_[0], item), dist;
        for(unsigned j = 1; j < mstack_.size(); ++j)
            if((dist = func_(mstack_[j])) < mindist) mindist = dist, i = j;
        return {i, mindist};
    }
    template<typename AItem>
    void add(const AItem &item, WT weight=1.) {
        auto [asn, mincost] = assign(item);
        auto cost = weight * mincost;
        auto gam = get_gamma();
        if(cost / f_ > urd_(rng_)) {
            mstack_.push(std::pair<Item, WT>{item, weight});
        } else {
            cost_ += cost;
            mstack_[asn].second += weight;
        }
        if(cost_ > gam * l_i_ || mstack_.size() > (gam - 1) * (1 + std::log(n_)) * k_) {
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
        ++n_;
        add(gen(), wgen());
    }
    template<typename Generator, typename WeightGen=UniformW<WT>>
    void process(Generator &gen, WeightGen &&wgen=WeightGen()) {
        WeightGen weight_gen(std::move(wgen));
        while(gen.size() /* maybe rename for easier interface? */ ) {
            process_step(gen, wgen);
        }
    }
    KServiceClusterer(Func func, unsigned k, size_t n, double alpha, uint64_t seed=std::rand()):
        func_(func), l_i_(1),
        alpha_(alpha), beta_(2. * alpha_ * alpha_  * get_cofl() + 2. * alpha_), k_(k), n_(n), i_(1)
    {
        rng_.seed(seed);
    }
};

template<typename Item, typename Func, template<typename> class WeightGen=UniformW, typename FT=double>
auto make_kservice_clusterer(Func func, unsigned k, size_t n, double alpha, bool uniform_weighting=is_uniform_weighting<WeightGen<FT>>::value) {
    std::fprintf(stderr, "Is uniform ? %d\n", uniform_weighting);
    return KServiceClusterer<Item, Func, FT>(func, k, n, alpha);
}

template<typename Item, template<typename> class WeightGen=UniformW, typename FT=double>
auto make_online_kmedian_clusterer(unsigned k, size_t n, bool uniform_weighting=is_uniform_weighting<WeightGen<FT>>::value) {
    blz::L1Norm func;
    return make_kservice_clusterer<Item, blz::L1Norm, FT>(blz::L1Norm(), k, n, 1., uniform_weighting);
}
template<typename Item, template<typename> class WeightGen=UniformW, typename FT=double>
auto make_online_kmeans_clusterer(unsigned k, size_t n, bool uniform_weighting=is_uniform_weighting<WeightGen<FT>>::value) {
    blz::sqrL2Norm func;
    return make_kservice_clusterer<Item, blz::sqrL2Norm, FT>(blz::sqrL2Norm(), k, n, 2., uniform_weighting);
}
template<typename Item, template<typename> class WeightGen=UniformW, typename FT=double>
auto make_online_l2_clusterer(unsigned k, size_t n, bool uniform_weighting=is_uniform_weighting<WeightGen<FT>>::value) {
    blz::L2Norm func;
    return make_kservice_clusterer<Item, blz::L2Norm, FT>(blz::L2Norm(), k, n, 1., uniform_weighting);
}

} // namespace streaming
} // namespace minocore

#endif /* FGC_STREAMING_H__ */
