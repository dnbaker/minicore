#pragma once
#include "minicore/utility.h"
#include "minicore/optim.h"
#include "minicore/dist/distance.h"

namespace minicore {


template<typename FT>
using CType = blz::DV<FT, blz::rowVector>;

struct SumOpts {

    bool load_csr = false, transpose_data = false, load_blaze = false;
    bool use_exponential_skips = false; // Use exponential skips if set; otherwise, use SIMD reservoir sampling
    dist::DissimilarityMeasure dis = dist::JSD;
    dist::Prior prior = dist::NONE;
    double gamma = 1.;
    double eps = 1e-9;
    unsigned k = 10;
    size_t coreset_samples = 1'000'000uL;
    uint64_t seed = 0;
    unsigned extra_sample_tries = 1;
    unsigned lloyd_max_rounds = 1000;
    unsigned sampled_number_coresets = 100;
    int lspp = 0;
    unsigned fl_b = -1;
    coresets::SensitivityMethod sm = coresets::BFL;
    bool soft = false;
    bool discrete_metric_search = false; // Search in metric solver before performing EM
    double outlier_fraction = 0.;
    // If nonzero, performs KMC2 with m kmc2_rounds as chain length
    // Otherwise, performs standard D2 sampling
    size_t kmc2_rounds = 0;
    std::shared_ptr<util::TimeStamper> stamper_;
    std::string to_string() const {
        constexpr const char *fmt = "K: %d. Construction method: %d. Seed:%zu. gamma: %g. prior: %d/%s."
                                    "extrasampletries: %d. coreset size: %zu. eps: %g. dis: %d/%s. %s clustering.%s";
        const char *dmsstr = discrete_metric_search ? "Use discrete metric search": "";
        const char *softstr = soft ? "soft": "hard";
        int nb = std::snprintf(nullptr, 0, fmt, k, (int)sm, size_t(seed), gamma, (int)prior, dist::prior2desc(prior), extra_sample_tries, coreset_samples, eps, (int)dis, dist::detail::prob2str(dis), softstr, dmsstr);
        std::string buf(nb, '\0');
        std::sprintf(buf.data(), fmt, k, (int)sm, size_t(seed), gamma, (int)prior, dist::prior2desc(prior), extra_sample_tries, coreset_samples, eps, (int)dis, dist::detail::prob2str(dis), softstr, dmsstr);
        std::string ret(buf);
        if(kmc2_rounds) {
            ret += "kmc2_rounds: ";
            ret += std::to_string(kmc2_rounds);
            ret += ".";
        }
        return buf;
    }
    SumOpts(SumOpts &&) = default;
    SumOpts(const SumOpts &) = delete;
    SumOpts() {}
    SumOpts(dist::DissimilarityMeasure measure, unsigned k=10, double prior_value=0., coresets::SensitivityMethod sm=coresets::BFL,
            double outlier_fraction=0., size_t max_rounds=1000, bool soft=false, size_t kmc2_rounds=0): dis(measure), prior(prior_value == 0. ? dist::NONE: prior_value == 1. ? dist::DIRICHLET: dist::GAMMA_BETA), gamma(prior_value), k(k),
            lloyd_max_rounds(max_rounds),
            sm(sm),
            soft(soft),
            outlier_fraction(outlier_fraction),
            kmc2_rounds(kmc2_rounds)
    {
    }
    SumOpts(int measure, unsigned k=10, double prior_value=0., std::string sm="BFL",
            double outlier_fraction=0., size_t max_rounds=1000, bool soft=false, size_t kmc2_rounds=0)
            : SumOpts((dist::DissimilarityMeasure)measure, k, prior_value, coresets::str2sm(sm), outlier_fraction, max_rounds, soft, kmc2_rounds) {}
    SumOpts(int measure, unsigned k=10, double prior_value=0., int smi=static_cast<int>(minicore::coresets::BFL),
            double outlier_fraction=0., size_t max_rounds=1000, bool soft=false, size_t kmc2_rounds=0)
        : SumOpts((dist::DissimilarityMeasure)measure, k, prior_value, static_cast<coresets::SensitivityMethod>(smi), outlier_fraction, max_rounds, soft, kmc2_rounds) {}
    SumOpts(std::string msr, unsigned k=10, double prior_value=0., int smi=static_cast<int>(minicore::coresets::BFL),
            double outlier_fraction=0., size_t max_rounds=1000, bool soft=false, size_t kmc2_rounds=0)
           : SumOpts(dist::str2msr(msr), k, prior_value, static_cast<coresets::SensitivityMethod>(smi), outlier_fraction, max_rounds, soft, kmc2_rounds) {}
    SumOpts(std::string msr, unsigned k=10, double prior_value=0., std::string sm="BFL", double outlier_fraction=0., size_t max_rounds=1000, bool soft=false, size_t kmc2_rounds=0)
        : SumOpts(dist::str2msr(msr), k, prior_value, coresets::str2sm(sm), outlier_fraction, max_rounds, soft, kmc2_rounds) {}
};




/*
 * get_initial_centers:
 * Samples points in proportion to their cost, as evaluated by norm(x, y).
 * For squared L2 distance, mu-similar Bregman divergences, and some other functions,
 * this produces an log(k)-approximate solution.
 *
 * TODO: perform 1 or 2 rounds of EM before saving costs, which might be a better heuristic
 */
template<typename Matrix, typename RNG, typename Norm=blz::sqrL2Norm, typename WeightT=const double>
auto get_initial_centers(const Matrix &matrix, RNG &rng,
                         unsigned k, unsigned kmc2_rounds, int lspp, bool use_exponential_skips, const Norm &norm, WeightT *const weights=static_cast<WeightT *>(nullptr))
{
    constexpr bool is_dense_matrix = blaze::IsDenseMatrix_v<Matrix>;
    using FT = double;
    const size_t nr = matrix.rows();
    std::vector<uint32_t> indices, asn;
    blz::DV<FT> costs(nr);
    const auto oracle = [&](auto xind, auto yind) ALWAYS_INLINE {
        return norm(row(matrix, xind, blz::unchecked), row(matrix, yind, blz::unchecked));
    };
    if(kmc2_rounds > 0) {
        std::fprintf(stderr, "[%s] Performing kmc\n", __func__);
        indices = coresets::kmc2(oracle, rng, nr, k, kmc2_rounds);
#ifndef NDEBUG
        std::fprintf(stderr, "Got indices of size %zu\n", indices.size());
        for(const auto idx: indices) if(idx > nr) std::fprintf(stderr, "idx %zu > max %zu\n", size_t(idx), nr);
#endif
        // Return distance from item at reference i to item at j
        auto [oasn, ncosts] = coresets::get_oracle_costs(oracle, nr, indices);
        costs = std::move(ncosts);
        asn.assign(oasn.data(), oasn.data() + oasn.size());
    } else {
        std::fprintf(stderr, "[%s] Performing kmeans++\n", __func__);
        std::vector<FT> fcosts;
        std::tie(indices, asn, fcosts) = coresets::kmeanspp(oracle, rng, nr, k, weights, lspp, use_exponential_skips, !is_dense_matrix);
        //indices = std::move(initcenters);
        std::copy(fcosts.data(), fcosts.data() + fcosts.size(), costs.data());
    }
    assert(*std::max_element(indices.begin(), indices.end()) < nr);
    return std::make_tuple(indices, asn, costs);
}

template<typename Matrix, typename RNG, typename Norm=blz::sqrL2Norm, typename WeightT=const double>
auto repeatedly_get_initial_centers(const Matrix &matrix, RNG &rng,
                                    unsigned k, unsigned kmc2_rounds, int ntimes, int lspp=0, bool use_exponential_skips=false, const Norm &norm=Norm(),
                                    WeightT *const weights=static_cast<WeightT *>(nullptr))
{
    using FT = double;
    auto [idx,asn,costs] = get_initial_centers(matrix, rng, k, kmc2_rounds, lspp, use_exponential_skips, norm, weights);
    auto tcost = blz::sum(costs);
    for(;--ntimes > 0;) {
        auto [_idx,_asn,_costs] = get_initial_centers(matrix, rng, k, kmc2_rounds, lspp, use_exponential_skips, norm, weights);
        auto ncost = blz::sum(_costs);
        if(ncost < tcost) {
            std::fprintf(stderr, "%g->%g: %g\n", tcost, ncost, tcost - ncost);
            std::tie(idx, asn, costs, tcost) = std::move(std::tie(_idx, _asn, _costs, ncost));
        }
    }
    CType<FT> modcosts(costs.size());
    std::copy(costs.begin(), costs.end(), modcosts.data());
    return std::make_tuple(idx, asn, modcosts); // Return a blaze vector
}


template<typename MT, bool SO, typename FT=double>
auto m2d2(blaze::Matrix<MT, SO> &sm, const SumOpts &opts, FT *weights=nullptr)
{
    blz::DV<FT, blz::rowVector> pc(1), *pcp = &pc;
    if(opts.prior == dist::DIRICHLET) pc[0] = 1.;
    else if(opts.prior == dist::GAMMA_BETA) pc[0] = opts.gamma;
    else if(opts.prior == dist::NONE)
        pcp = nullptr;
    auto app = jsd::make_probdiv_applicator(*sm, opts.dis, opts.prior, pcp);
    wy::WyRand<uint64_t, 2> rng(opts.seed);
    auto [centers, asn, costs] = jsd::make_kmeanspp(app, opts.k, opts.seed, weights, opts.use_exponential_skips);
    auto csum = blz::sum(costs);
    for(unsigned i = 0; i < opts.extra_sample_tries; ++i) {
        auto [centers2, asn2, costs2] = jsd::make_kmeanspp(app, opts.k, opts.seed, weights);
        if(auto csum2 = blz::sum(costs2); csum2 < csum) {
            std::tie(centers, asn, costs, csum) = std::move(std::tie(centers2, asn2, costs2, csum2));
        }
    }
    if(opts.lspp) {
        localsearchpp_rounds(app, rng, costs, centers, asn, costs.size(), opts.lspp, weights);
    }
    CType<FT> modcosts(costs.size());
    std::copy(costs.begin(), costs.end(), modcosts.begin());
    return std::make_tuple(centers, asn, costs);
}

template<typename FT, bool SO>
auto m2greedysel(blaze::Matrix<FT, SO> &sm, const SumOpts &opts)
{
    blz::DV<blaze::ElementType_t<FT>, blz::rowVector> pc(1), *pcp = &pc;
    if(opts.prior == dist::DIRICHLET) pc[0] = 1.;
    else if(opts.prior == dist::GAMMA_BETA) pc[0] = opts.gamma;
    else if(opts.prior == dist::NONE)
        pcp = nullptr;
    auto app = jsd::make_probdiv_applicator(*sm, opts.dis, opts.prior, pcp);
    wy::WyRand<uint64_t, 2> rng(opts.seed);
    if(opts.outlier_fraction) {
        return coresets::kcenter_greedy_2approx_outliers_costs<decltype(app), double, uint64_t>(
            app, app.size(), rng, opts.k,
            /*eps=*/1.5, opts.outlier_fraction
        );
    } else return coresets::kcenter_greedy_2approx_costs<decltype(app), double, uint64_t>(app, app.size(), opts.k, rng);
}

template<typename VT, typename IndicesT, typename IndPtrT>
std::pair<std::vector<uint64_t>, std::vector<double>> m2greedysel(util::CSparseMatrix<VT, IndicesT, IndPtrT> &matrix, const SumOpts &opts)
{
    using FT = std::conditional_t<(sizeof(VT) <= 4), float, double>;
    blz::DV<FT ,blz::rowVector> pc(1, 0.);
    if(opts.prior == dist::DIRICHLET) pc[0] = 1.;
    else if(opts.prior == dist::GAMMA_BETA) pc[0] = opts.gamma;
    const FT prior_sum = pc[0] * matrix.columns();
    blz::DV<FT> rsums = util::sum<blaze::rowwise>(matrix);
    auto oracle = [&](size_t x, size_t y) {
        return cmp::msr_with_prior(opts.dis, row(matrix, y, blz::unchecked), row(matrix, x, blz::unchecked), pc, prior_sum, rsums[y], rsums[x]);
    };
    wy::WyRand<uint64_t, 2> rng(opts.seed);
    if(opts.outlier_fraction) {
        return coresets::kcenter_greedy_2approx_outliers_costs<decltype(oracle), double, uint64_t>(
            oracle, matrix.rows(), rng, opts.k,
            /*eps=*/1.5, opts.outlier_fraction
        );
    } else return coresets::kcenter_greedy_2approx_costs<decltype(oracle), double, uint64_t>(oracle, matrix.rows(), opts.k, rng);
}

template<typename VT, typename IndicesT, typename IndPtrT, typename FT=double>
auto m2d2(const util::CSparseMatrix<VT, IndicesT, IndPtrT> &matrix, const SumOpts &opts, FT *weights=static_cast<FT *>(nullptr))
{
    blz::DV<FT, blz::rowVector> pc(1);
    if(opts.prior == dist::DIRICHLET) pc[0] = 1.;
    else if(opts.prior == dist::GAMMA_BETA) pc[0] = opts.gamma;
    const FT prior_sum = pc[0] * matrix.columns();
    blz::DV<FT> rsums = util::sum<blaze::rowwise>(matrix);
    auto oracle = [&](size_t x, size_t y) {
        return cmp::msr_with_prior(opts.dis, row(matrix, y, blz::unchecked), row(matrix, x, blz::unchecked), pc, prior_sum, rsums[y], rsums[x]);
    };
    wy::WyRand<uint64_t, 2> rng(opts.seed);
    auto [centers, asn, costs] = coresets::kmeanspp(oracle, rng, matrix.rows(), opts.k, weights, opts.use_exponential_skips);
    auto csum = blz::sum(costs);
    for(unsigned i = 0; i < opts.extra_sample_tries; ++i) {
        auto [centers2, asn2, costs2] = coresets::kmeanspp(oracle, rng, matrix.rows(), opts.k, weights, opts.use_exponential_skips);
        if(auto csum2 = blz::sum(costs2); csum2 < csum) {
            std::tie(centers, asn, costs, csum) = std::move(std::tie(centers2, asn2, costs2, csum2));
        }
    }
    if(opts.lspp > 0) {
        localsearchpp_rounds(oracle, rng, costs, centers, asn, costs.size(), opts.lspp, weights);
    }
    CType<FT> modcosts(costs.size());
    std::copy(costs.begin(), costs.end(), modcosts.begin());
    return std::make_tuple(centers, asn, costs);
}

} // namespace minicore
