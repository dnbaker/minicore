#pragma once
#include "minocore/minocore.h"
namespace minocore {
struct Opts {
    size_t kmc2_rounds = 0;
    bool load_csr = false, transpose_data = false, load_blaze = false;
    dist::DissimilarityMeasure dis = dist::JSD;
    dist::Prior prior = dist::DIRICHLET;
    double gamma = 1.;
    double eps = 1e-9;
    unsigned k = 10;
    size_t coreset_size = 1000;
    uint64_t seed = 0;
    unsigned extra_sample_tries = 10;
    unsigned lloyd_max_rounds = 1000;
    unsigned sampled_number_coresets = 100;
    coresets::SensitivityMethod sm = coresets::BFL;
    bool soft = false;
    // If nonzero, performs KMC2 with m kmc2_rounds as chain length
    // Otherwise, performs standard D2 sampling
};


template<typename MT, bool SO, typename RNG>
auto get_initial_centers(blaze::Matrix<MT, SO> &matrix, RNG &rng,
                         unsigned k, unsigned kmc2rounds) {
    using FT = blaze::ElementType_t<MT>;
    const size_t nr = (~matrix).rows(), nc = (~matrix).columns();
    std::vector<uint32_t> indices, asn;
    blz::DV<FT> costs(nr);
    if(kmc2rounds) {
        std::fprintf(stderr, "Performing kmc\n");
        indices = coresets::kmc2(matrix, rng, k, kmc2rounds, blz::sqrL2Norm());
        auto oracle = [&](size_t i, size_t j) {
            // Return distance from item at reference i to item at j
            return blz::sqrNorm(row(~matrix, i, blz::unchecked) - row(~matrix, j, blz::unchecked));
        };
        auto [oasn, ncosts] = coresets::get_oracle_costs(oracle, nr, indices);
        costs = std::move(ncosts);
        asn.assign(oasn.data(), oasn.data() + oasn.size());
    } else {
        std::fprintf(stderr, "Performing kmeanspp\n");
        std::vector<FT> fcosts;
        std::tie(indices, asn, fcosts) = coresets::kmeanspp(matrix, rng, k, blz::sqrL2Norm());
        //indices = std::move(initcenters);
        std::copy(fcosts.data(), fcosts.data() + fcosts.size(), costs.data());
    }
    assert(*std::max_element(indices.begin(), indices.end()) < nr);
    return std::make_tuple(indices, asn, costs);
}

template<typename MT, bool SO, typename RNG>
auto repeatedly_get_initial_centers(blaze::Matrix<MT, SO> &matrix, RNG &rng,
                         unsigned k, unsigned kmc2rounds, unsigned ntimes) {
    using FT = blaze::ElementType_t<MT>;
    if(ntimes > 0) --ntimes;
    auto [idx,asn,costs] = get_initial_centers(matrix, rng, k, kmc2rounds);
    auto tcost = blz::sum(costs);
    for(;ntimes--;) {
        auto [_idx,_asn,_costs] = get_initial_centers(matrix, rng, k, kmc2rounds);
        auto ncost = blz::sum(_costs);
        if(ncost < tcost) {
            std::fprintf(stderr, "%g->%g: %g\n", tcost, ncost, tcost - ncost);
            std::tie(idx, asn, costs, tcost) = std::move(std::tie(_idx, _asn, _costs, ncost));
        }
    }
    CType<FT> modcosts(costs.size());
    std::copy(costs.begin(), costs.end(), modcosts.begin());
    return std::make_tuple(idx, asn, modcosts); // Return a blaze vector
}
} // namespace minocore
