#ifndef L2_DETAIL_H__
#define L2_DETAIL_H__
#include "mtx2cs.h"
#include <vector>
#include "minicore/util/blaze_adaptor.h"

namespace minicore {

template<typename FT>
std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, blz::DV<FT, blz::rowVector>>
get_jv_centers(blz::SM<FT> &mat, unsigned k, unsigned maxiter, double, uint64_t);


template<typename FT>
std::tuple<std::vector<blz::DV<FT, blz::rowVector>>, std::vector<uint32_t>, blz::DV<FT, blz::rowVector>>
l2_sum_core(blz::SM<FT> &mat, std::string out, SumOpts &opts) {
    wy::WyRand<uint64_t, 2> rng(opts.seed);
    std::vector<uint32_t> indices, asn;
    blz::DV<FT, blz::rowVector> costs;
    if(opts.discrete_metric_search) { // Use EM to solve instead of JV
        std::tie(indices, asn, costs) = get_jv_centers(mat, opts.k, opts.lloyd_max_rounds, opts.eps, opts.seed);
    } else {
        std::tie(indices, asn, costs) = repeatedly_get_initial_centers(mat, rng, opts.k, opts.kmc2_rounds, opts.extra_sample_tries, opts.lspp, opts.use_exponential_skips, blz::L2Norm());
    }
    std::vector<blz::DV<FT, blz::rowVector>> centers(opts.k);
    { // write selected initial points to file
        std::ofstream ofs(out + ".initial_points");
        for(size_t i = 0; i < indices.size(); ++i) {
            ofs << indices[i] << ',';
        }
        ofs << '\n';
    }
    std::unique_ptr<uint32_t[]> counts(new uint32_t[opts.k]());
    OMP_PFOR
    for(size_t i = 0; i < mat.rows(); ++i) {
        OMP_ATOMIC
        ++counts[asn[i]];
    }
    OMP_PFOR
    for(unsigned i = 0; i < opts.k; ++i) {
        centers[i] = row(mat, indices[i]);
        std::fprintf(stderr, "Center %u initialized by index %u, %u supporters and has sum of %g\n", i, indices[i], counts[i], blz::sum(centers[i]));
    }
    FT tcost = blz::sum(costs), firstcost = tcost;
    size_t iternum = 0;
    OMP_ONLY(std::unique_ptr<std::mutex[]> mutexes(new std::mutex[opts.k]);)
    for(;;) {
        std::fprintf(stderr, "[Iter %zu] Cost: %g\n", iternum, tcost);
        center_setup:
        blz::SmallArray<uint32_t, 16> sa;
        OMP_PFOR
        for(size_t i = 0; i < opts.k; ++i) {
            auto submat = blz::rows(mat,  blz::functional::indices_if([&](auto x) {return asn[x] == i;}, asn.size()));
            if(!submat.rows())
                sa.pushBack(i);
            else
                blz::geomedian(submat, centers[i]);
        }
        // Set centers
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
                    auto c = blz::l2Norm(centers[idx] - row(mat, i, blz::unchecked));
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
            costs[i] = blz::l2Norm(lhr - centers[0]);
            for(unsigned j = 1; j < opts.k; ++j)
                if(auto v = blz::l2Norm(lhr - centers[j]);
                   v < costs[i]) costs[i] = v, asn[i] = j;
            assert(asn[i] < opts.k);
        }
        auto newcost = blz::sum(costs);
        std::fprintf(stderr, "newcost: %.16g. Cost changed by %0.12g at iter %zu\n", newcost, newcost - tcost, iternum);
        if(std::abs(newcost - tcost) < opts.eps * firstcost) {
            break;
        }
        tcost = newcost;
        if(++iternum >= opts.lloyd_max_rounds) {
            break;
        }
    }
    return std::make_tuple(centers, asn, costs);
}

template<typename FT>
std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, blz::DV<FT, blz::rowVector>>
get_jv_centers(blz::SM<FT> &mat, unsigned k, unsigned maxiter, double eps, uint64_t seed) {
    auto start = std::chrono::high_resolution_clock::now();
    diskmat::PolymorphicMat<FT> distmat(mat.rows(), mat.rows());
    auto &dm = ~distmat;
    size_t np = dm.rows();
    for(size_t i = 0; i < mat.rows(); ++i) {
        auto r = row(dm, i, blz::unchecked);
        auto dr = row(mat, i, blz::unchecked);
        r[i] = 0;
        OMP_PFOR
        for(size_t j = i + 1; j < mat.rows(); ++j) {
            dm(j, i) = r[j] = blz::l2Norm(dr - row(mat, j, blz::unchecked));
        }
#ifndef NDEBUG
        if(unlikely(i % 16 == 0)) {
            std::fprintf(stderr, "Computed %zu/%zu rows in %gms\n", i, mat.rows(), util::timediff2ms(start, std::chrono::high_resolution_clock::now()));
        }
#endif
    }
    auto distmattime = std::chrono::high_resolution_clock::now();
    // Run JV
    std::fprintf(stderr, "[get_jv_centers:%s:%d] Time to compute distance matrix: %gms\n", __FILE__, __LINE__, util::timediff2ms(start, distmattime));
#if USE_JV_TOO
    std::fprintf(stderr, "[get_ms_centers_l1:%s:%d] Time to compute distance matrix: %gms\n", __FILE__, __LINE__, util::timediff2ms(start, distmattime));
    auto solver = jv::make_jv_solver(dm);
    auto [fac, asn] = solver.kmedian(k, maxiter);
    std::fprintf(stderr, "initial centers: "); for(const auto f: fac) std::fprintf(stderr, ",%d", f); std::fputc('\n', stderr);
    shared::sort(fac.begin(), fac.end());
#else
    std::vector<uint32_t> fac;
#endif
    // Local search from that solution (if JV), otherwise from k-center approx
    auto lsearcher = make_kmed_lsearcher(dm, k, eps, seed + maxiter);
    lsearcher.lazy_eval_ = 2;
#if USE_JV_TOO
    lsearcher.assign_centers(fac.begin(), fac.end());
#endif
    lsearcher.run();
    fac.assign(lsearcher.sol_.begin(), lsearcher.sol_.end());
    auto fbeg = &fac[0], fend = &fac[k];
    blz::DV<FT, blz::rowVector> costs(np);
    std::vector<uint32_t> asnret(np);
    OMP_PFOR
    for(size_t i = 0; i < np; ++i) {
        auto r = row(dm, i, blz::unchecked);
        auto assignment = std::min_element(fbeg, fend, [&](auto x, auto y) {return r[x] < r[y];}) - fbeg;
        asnret[i] = assignment;
        costs[i] = r[assignment];
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "jvcost: %0.12g in %gms\n", blz::sum(costs), util::timediff2ms(start, stop));
    return std::make_tuple(fac, asnret, costs);
}

} // namespace minicore
#endif
