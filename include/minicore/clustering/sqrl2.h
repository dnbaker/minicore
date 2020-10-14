#ifndef SQRL2_DETAIL_H__
#define SQRL2_DETAIL_H__
#include "mtx2cs.h"

namespace minicore {

template<typename FT>
auto
kmeans_sum_core(blz::SM<FT> &mat, std::string DBG_ONLY(out), SumOpts &opts) {
    wy::WyRand<uint64_t, 2> rng(opts.seed);
    std::vector<uint32_t> indices, asn;
    blz::DV<FT, blz::rowVector> costs;
    if(opts.stamper_) opts.stamper_->add_event("Get initial centers");
    std::tie(indices, asn, costs) = repeatedly_get_initial_centers(mat, rng, opts.k, opts.kmc2_rounds, opts.extra_sample_tries);
    std::vector<blz::DV<FT, blz::rowVector>> centers(opts.k);
#ifndef NDEBUG
    { // write selected initial points to file
        if(opts.stamper_) opts.stamper_->add_event("Save initial points to disk");
        std::ofstream ofs(out + ".initial_points");
        for(size_t i = 0; i < indices.size(); ++i) {
            ofs << indices[i] << ',';
        }
        ofs << '\n';
    }
#endif
    if(opts.stamper_) opts.stamper_->add_event("Set initial centers");
    OMP_PFOR
    for(unsigned i = 0; i < opts.k; ++i) {
        centers[i] = row(mat, indices[i]);
#ifndef NDEBUG
        std::fprintf(stderr, "Center %u initialized by index %u and has sum of %g\n", i, indices[i], blz::sum(centers[i]));
#endif
    }
    if(opts.stamper_) opts.stamper_->add_event("Setup locks + counts");
    FT tcost = blz::sum(costs), firstcost = tcost;
    size_t iternum = 0;
    OMP_ONLY(std::unique_ptr<std::mutex[]> mutexes(new std::mutex[opts.k]);)
    std::unique_ptr<uint32_t[]> counts(new uint32_t[opts.k]);
    if(opts.stamper_) opts.stamper_->add_event("Optimize");
    for(;;) {
        std::fprintf(stderr, "[Iter %zu] Cost: %g\n", iternum, tcost);
        // Set centers
        center_setup:
        for(auto &c: centers) c = 0.;
        std::fill_n(counts.get(), opts.k, 0u);
        OMP_PFOR
        for(size_t i = 0; i < mat.rows(); ++i) {
            auto myasn = asn[i];
            OMP_ONLY(std::lock_guard<std::mutex> lock(mutexes[myasn]);)
            centers[myasn] += row(mat, i, blaze::unchecked);
            OMP_ATOMIC
            ++counts[myasn];
        }
        blz::SmallArray<uint32_t, 16> sa;
        for(unsigned i = 0; i < opts.k; ++i) {
            if(counts[i]) {
                centers[i] /= counts[i];
            } else {
                sa.pushBack(i);
            }
        }
        if(sa.size()) {
            for(unsigned i = 0; i < sa.size(); ++i) {
                const auto idx = sa[i];
                blz::DV<FT> probs(mat.rows());
                FT *pd = probs.data(), *pe = pd + probs.size();
                std::partial_sum(costs.begin(), costs.end(), pd);
                std::uniform_real_distribution<double> dist;
                std::ptrdiff_t found = std::lower_bound(pd, pe, dist(rng) * pe[-1]) - pd;
                centers[idx] = row(mat, found);
                for(size_t i = 0; i < mat.rows(); ++i) {
                    auto c = blz::sqrNorm(centers[idx] - row(mat, i, blz::unchecked));
                    if(c < costs[i]) {
                        asn[i] = idx;
                        costs[i] = c;
                    }
                }
            }
            goto center_setup;
        }
        std::fill(asn.begin(), asn.end(), 0);
        OMP_PFOR
        for(size_t i = 0; i < mat.rows(); ++i) {
            auto lhr = row(mat, i, blaze::unchecked);
            asn[i] = 0;
            costs[i] = blz::sqrNorm(lhr - centers[0]);
            for(unsigned j = 1; j < opts.k; ++j)
                if(auto v = blz::sqrNorm(lhr - centers[j]);
                   v < costs[i]) costs[i] = v, asn[i] = j;
            assert(asn[i] < opts.k);
        }
        auto newcost = blz::sum(costs);
        std::fprintf(stderr, "newcost: %g. Cost changed by %g at iter %zu\n", newcost, newcost - tcost, iternum);
        if(std::abs(newcost - tcost) < opts.eps * firstcost) {
            break;
        }
        tcost = newcost;
        if(++iternum >= opts.lloyd_max_rounds) {
            break;
        }
    }
    std::fprintf(stderr, "Completed: clustering\n");
    return std::make_tuple(centers, asn, costs);
}

} // namespace minicore

#endif


