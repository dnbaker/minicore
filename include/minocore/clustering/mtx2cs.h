#pragma once
#include "minocore/utility.h"
#include "minocore/optim.h"

namespace minocore {


template<typename FT>
using CType = blz::DV<FT, blz::rowVector>;

namespace dist = ::blz::distance;

struct Opts {
    size_t kmc2_rounds = 0;
    // If nonzero, performs KMC2 with m kmc2_rounds as chain length
    // Otherwise, performs standard D2 sampling

    bool load_csr = false, transpose_data = false, load_blaze = false;
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
    coresets::SensitivityMethod sm = coresets::BFL;
    bool soft = false;
    bool discrete_metric_search = false; // Search in metric solver before performing EM
    std::shared_ptr<util::TimeStamper> stamper_;
    std::string to_string() const {
        constexpr const char *fmt = "K: %d. Construction method: %d. Seed:%zu. gamma: %g. prior: %d/%s."
                                    "extrasampletries: %d. coreset size: %zu. eps: %g. dis: %d/%s. %s clustering.%s";
        const char *dmsstr = discrete_metric_search ? "Use discrete metric search": "";
        const char *softstr = soft ? "soft": "hard";
        int nb = std::snprintf(nullptr, 0, fmt, k, (int)sm, size_t(seed), gamma, (int)prior, dist::prior2desc(prior), extra_sample_tries, coreset_samples, eps, (int)dis, dist::detail::prob2str(dis), softstr, dmsstr);
        std::string buf(nb, '\0');
        std::sprintf(buf.data(), fmt, k, (int)sm, size_t(seed), gamma, (int)prior, dist::prior2desc(prior), extra_sample_tries, coreset_samples, eps, (int)dis, dist::detail::prob2str(dis), softstr, dmsstr);
        return buf;
    }
    Opts(Opts &&) = default;
    Opts(const Opts &) = delete;
    Opts() {}
};


/*
 * TODO: perform 1 or 2 rounds of EM before saving costs, which might be a better heuristic
 */
template<typename MT, bool SO, typename RNG, typename Norm=blz::sqrL2Norm>
auto get_initial_centers(blaze::Matrix<MT, SO> &matrix, RNG &rng,
                         unsigned k, unsigned kmc2rounds, const Norm &norm) {
    using FT = blaze::ElementType_t<MT>;
    const size_t nr = (~matrix).rows();
    std::vector<uint32_t> indices, asn;
    blz::DV<FT> costs(nr);
    if(kmc2rounds) {
        std::fprintf(stderr, "Performing kmc\n");
        indices = coresets::kmc2(matrix, rng, k, kmc2rounds, norm);
        // Return distance from item at reference i to item at j
        auto oracle = [&](size_t i, size_t j) {
            return norm(row(~matrix, i, blz::unchecked), row(~matrix, j, blz::unchecked));
        };
        auto [oasn, ncosts] = coresets::get_oracle_costs(oracle, nr, indices);
        costs = std::move(ncosts);
        asn.assign(oasn.data(), oasn.data() + oasn.size());
    } else {
        std::fprintf(stderr, "Performing kmeanspp\n");
        std::vector<FT> fcosts;
        std::tie(indices, asn, fcosts) = coresets::kmeanspp(matrix, rng, k, norm);
        //indices = std::move(initcenters);
        std::copy(fcosts.data(), fcosts.data() + fcosts.size(), costs.data());
    }
    assert(*std::max_element(indices.begin(), indices.end()) < nr);
    return std::make_tuple(indices, asn, costs);
}

template<typename MT, bool SO, typename RNG, typename Norm=blz::sqrL2Norm>
auto repeatedly_get_initial_centers(blaze::Matrix<MT, SO> &matrix, RNG &rng,
                         unsigned k, unsigned kmc2rounds, unsigned ntimes, const Norm &norm=Norm()) {
    using FT = blaze::ElementType_t<MT>;
    auto [idx,asn,costs] = get_initial_centers(matrix, rng, k, kmc2rounds, norm);
    auto tcost = blz::sum(costs);
    for(;--ntimes;) {
        auto [_idx,_asn,_costs] = get_initial_centers(matrix, rng, k, kmc2rounds, norm);
        auto ncost = blz::sum(_costs);
        if(ncost < tcost) {
            std::fprintf(stderr, "%g->%g: %g\n", tcost, ncost, tcost - ncost);
            std::tie(idx, asn, costs, tcost) = std::move(std::tie(_idx, _asn, _costs, ncost));
        }
    }
    blz::DV<FT, blz::rowVector> modcosts(costs.size());
    std::copy(costs.begin(), costs.end(), modcosts.begin());
    return std::make_tuple(idx, asn, modcosts); // Return a blaze vector
}

} // namespace minocore
