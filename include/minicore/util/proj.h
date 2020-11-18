#ifndef MC_L1_PROJ_H
#define MC_L1_PROJ_H

#include "blaze/Math.h"
#include "macros.h"

namespace minicore {

// Implemements projection to L1 ball
// Algorithm 2 in D Sculley, Web-scale K-means clustering


template<typename VT, bool TF>
INLINE double sum_min_theta(const blaze::DenseVector<VT, TF> &vector, double theta) {
    return  blaze::sum(blaze::max(0., blaze::abs(*vector) - theta));
}
template<typename VT, bool TF>
INLINE double sum_min_theta(const blaze::SparseVector<VT, TF> &vector, double theta) {
    return std::accumulate((*vector).begin(),(*vector).end(), 0., [theta](double sum, const auto &pair) {return sum + std::max(0., pair.value() - theta);});
}


// Perform L1-ball projection
template<typename VT, bool TF>
INLINE void set_min_theta(blaze::DenseVector<VT, TF> &vector, double theta) {
    for(size_t i = 0; i < (*vector).size(); ++i) {
        auto &v = (*vector)[i];
        if(v > theta) v -= theta;
        else if(v < -theta) v += theta;
        else v = 0.;
    }
}
template<typename VT, bool TF>
INLINE void set_min_theta(blaze::SparseVector<VT, TF> &vector, double theta) {
    (*vector).erase([theta](auto &x) {
        if(x < -theta)     x += theta;
        else if(x > theta) x -= theta;
        else               x = 0.;
        return x == 0.;
    });
}

template<typename VT, bool TF>
void eps_l1(blaze::Vector<VT, TF> &vector, double radius, double eps=1e-10) {
    auto &v = *vector;
    double l1n = blaze::l1Norm(v);
    if(l1n <= radius + eps) return;
    double lower = 0., current = l1n, upper = blaze::maxNorm(v);
    static constexpr bool is_sparse = blaze::IsSparseVector_v<VT>;
    while(current < radius || current > (radius * (1. + eps))) {
        double theta = .5 * upper + .5 * lower;
        current = sum_min_theta(v, theta);
        if(current <= theta) upper = theta;
                        else lower = theta;
    }
}

} // namespace minicore
#endif
