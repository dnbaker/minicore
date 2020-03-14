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
    FT rinv_, brinv_;
    JSDLSHasher(size_t nd, size_t nh, unsigned r, uint64_t seed=0) {
        if(seed == 0) seed = nd * nh + r;
        std::mt19937_64 mt(seed);
        std::normal_distribution<FT> gen;
        randproj_ = blz::generate(nh, nd, [&](size_t, size_t){return gen(mt);});
        boffsets_ = blz::generate(nh, [&](size_t){return FT(mt()) / mt.max();});
    }
    template<typename VT, bool SO>
    decltype(auto) hash(const blz::Vector<VT, SO> &input) {
        //return randproj_ * blz::sqrt(~input) + boffsets_;
        return blz::ceil(randproj_ * blz::sqrt(~input) + boffsets_);
    }
};

}

using hash::JSDLSHasher;

}

#endif /* FGC_HASH_H__ */
