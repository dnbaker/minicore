_Pragma("once")
#ifndef FGC_HASH_H__
#define FGC_HASH_H__
#include "blaze_adaptor.h"
#include <random>

namespace fgc {

namespace hash {

template<typename FT=double>
struct JSDLSHasher {
    // See https://papers.nips.cc/paper/9195-locality-sensitive-hashing-for-f-divergences-mutual-information-loss-and-beyond.pdf
    // for the function.
    // Note that this is an LSH for the JS Divergence, not the metric.
    blz::DM<FT> randproj_;
    blz::DV<FT> boffsets_;
    JSDLSHasher(size_t nd, size_t nh, double r, uint64_t seed=0) {
        if(seed == 0) seed = nd * nh + r;
        std::mt19937_64 mt(seed);
        std::normal_distribution<FT> gen;
        randproj_ = blz::generate(nh, nd, [&](size_t, size_t){return gen(mt);}) / r;
        boffsets_ = blz::generate(nh, [&](size_t){return FT(mt()) / mt.max();});
    }
    template<typename VT, bool SO>
    decltype(auto) hash(const blz::Vector<VT, SO> &input) {
        //return randproj_ * blz::sqrt(~input) + boffsets_;
        return blz::ceil(randproj_ * blz::sqrt(~input) + boffsets_);
    }
    template<typename VT, bool SO>
    decltype(auto) hash(const blz::Matrix<VT, SO> &input) {
        return blz::ceil(randproj_ * ~input + blz::expand(boffsets_, (~input).columns()));
    }
};

template<typename FT=double>
struct S2JSDLSHasher {
    // See S2JSD-LSH: A Locality-Sensitive Hashing Schema for Probability Distributions
    // https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14692
    // for the derivation
    // Note that this is an LSH for the JS Metric, not the JSD.
    blz::DM<FT> randproj_;
    blz::DV<FT> boffsets_;
    S2JSDLSHasher(size_t nd, size_t nh, double w, uint64_t seed=0) {
        if(seed == 0) seed = nd * nh  + w + 1. / w;
        std::mt19937_64 mt(seed);
        std::normal_distribution<FT> gen;
        randproj_ = blz::abs(blz::generate(nh, nd, [&](size_t, size_t){return gen(mt);}) * (4. / (w * w)));
        boffsets_ = blz::generate(nh, [&](size_t){return FT(mt() / 2) / mt.max();}) - 0.5;
    }
    template<typename VT, bool SO>
    decltype(auto) hash(const blz::Vector<VT, SO> &input) {
        return blz::floor(blz::sqrt(randproj_ * (~input) + 1.) + boffsets_);
    }
    template<typename VT, bool SO>
    decltype(auto) hash(const blz::Matrix<VT, SO> &input) {
        return blz::floor(blz::sqrt(randproj_ * (~input) + 1.) + blz::expand(boffsets_, (~input).columns()));
    }
};


}

using hash::JSDLSHasher;
using hash::S2JSDLSHasher;

}

#endif /* FGC_HASH_H__ */
