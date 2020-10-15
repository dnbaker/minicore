#pragma once
#ifndef FGC_GMM_H__
#define FGC_GMM_H__
#include "minicore/optim/kmeans.h"

namespace minicore {

template<typename FT=float, bool SO=blaze::rowMajor>
struct GMM {
    // Related: Laplacian
    unsigned                    k_;
    blaze::DynamicMatrix<FT, SO> mu_;
    blaze::DynamicMatrix<FT, SO> pi_;
    blaze::DynamicMatrix<FT, SO> pm_; // precision matrix
    std::vector<FT>    cached_det_;
    static constexpr double m_pi = 3.14159265358979323846;

    template<typename T>
    double logprob(const T &x, unsigned compnum) {
        auto ldiv = -0.5 * std::log(std::pow(2. * m_pi, this->dim()) * get_det(compnum));
        auto diff = x - submu(compnum);
        auto lnum = -0.5 * dot(diff, subprec(compnum) * trans(diff));
        return ldiv + lnum;
    }
    template<typename T, typename T2, typename=std::enable_if_t<!std::is_integral_v<T2>>>
    double logprob(const T &x, const T2 &assignments) {
        assert(assignments.size() == dim());
        std::vector<double> probs(assignments.size());
        double mv = std::numeric_limits<double>::min();
        for(size_t i = 0; i < probs.size(); ++i) {
            // Use logsumexp
            auto v = assignments[i] * std::exp(logprob(x, i));
            if(v > mv) {
                mv = v;
            }
            probs[i] = v;
        }
        auto ret = std::accumulate(probs.begin(), probs.end(), 0., [mv](auto cs, auto newv) {return cs + std::exp(newv - mv);});
        return mv + std::log(ret);
    }
    auto ncomp() const {return k_;}
    auto dim() const {return mu_.columns();}
    auto subcovar(unsigned compnum) {
        return submatrix(pi_, compnum * dim(), 0, dim(), dim());
    }
    auto submu(unsigned compnum) {
        return row(mu_, compnum);
    }
    auto subprec(unsigned compnum) {
        return submatrix(pm_, compnum * dim(), 0, dim(), dim());
    }
    void setprecision() {
        assert(mu_.rows() % mu_.columns() == 0);
        for(unsigned i = 0; i < k_; ++i) {
            subprec(i) = inv(declsym(subcovar(i)));
        }
    }
    auto get_det(unsigned compnum) {
        auto v = cached_det_[compnum];
        if(v == std::numeric_limits<FT>::max()) {
            v = det(declsym(subcovar(compnum)));
            cached_det_[compnum] = v;
        }
        return v;
    }
    auto &mu() {return mu_;}
    auto &pi() {return pi_;}
    GMM(unsigned k, size_t dim):
        k_(k), mu_(k, dim), pi_(k * dim, dim),
        pm_(pi_.rows(), pi_.columns()),
        cached_det_(k, std::numeric_limits<FT>::max())
    {
    }
};

} // minicore

#endif /* FGC_GMM_H__ */
