#ifndef L1_DETAIL_H__
#define L1_DETAIL_H__
#include "mtx2cs.h"
#include <vector>
#include "minicore/util/blaze_adaptor.h"

namespace minicore {


namespace clustering {


template<typename FT>
std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, blz::DV<FT, blz::rowVector>>
get_ms_centers_l1(blz::SM<FT> &mat, unsigned k, unsigned maxiter, double, uint64_t);

struct IndexCmp {
    template<typename T>
    bool operator()(const T x, const T y) const {return x->index() > y->index();}
    template<typename T, typename IT>
    bool operator()(const std::pair<T, IT> x, const std::pair<T, IT> y) const {
        return this->operator()(x.first, y.first);
    }
};

template<typename CI, typename IT=uint32_t>
struct IndexPQ: public std::priority_queue<std::pair<CI, IT>, std::vector<std::pair<CI, IT>>, IndexCmp> {
    IndexPQ(size_t nelem) {
        this->c.reserve(nelem);
    }
    auto &getc() {return this->c;}
    const auto &getc() const {return this->c;}
    auto getsorted() const {
        auto tmp = getc();
        std::sort(tmp.begin(), tmp.end(), this->comp);
        return tmp;
    }
};

template<typename FT>
std::tuple<std::vector<blz::DV<FT, blz::rowVector>>, std::vector<uint32_t>, blz::DV<FT, blz::rowVector>>
l1_sum_core(blz::SM<FT> &mat, std::string out, SumOpts &opts) {
    wy::WyRand<uint64_t, 2> rng(opts.seed);
    std::vector<uint32_t> indices, asn;
    blz::DV<FT, blz::rowVector> costs;
    if(opts.discrete_metric_search) { // Use EM to solve instead of JV
        std::tie(indices, asn, costs) = get_ms_centers_l1(mat, opts.k, opts.lloyd_max_rounds, opts.eps, opts.seed);
    } else {
        std::tie(indices, asn, costs) = repeatedly_get_initial_centers(mat, rng, opts.k, opts.kmc2_rounds, opts.extra_sample_tries, opts.lspp, blz::L1Norm());
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
        blz::SmallArray<uint32_t, 16> sa;
        std::vector<std::vector<uint32_t>> asns(opts.k);
        for(size_t i = 0; i < mat.rows(); ++i) {
            asns[asn[i]].push_back(i);
        }
        center_setup:
        OMP_PFOR
        for(uint32_t i = 0; i < opts.k; ++i) {
            auto &asn(asns[i]);
            const auto numasn = asn.size();
            assert(numasn == std::unordered_set<uint32_t>(asn.begin(), asn.end()).size());
            std::fprintf(stderr, "center %d has %zu assignments currently\n", i, numasn);
            switch(numasn) {
                case 0:
                OMP_CRITICAL {sa.pushBack(i);}
                continue;
                case 1: centers[i] = row(mat, asn[0]); continue;
                case 2: centers[i] = .5 * (row(mat, asn[1]) + row(mat, asn[0])); continue;
                default:;
            }
            auto med1ts = std::chrono::high_resolution_clock::now();
            coresets::l1_median(blz::rows(mat, asn.data(), asn.size()), centers[i]);
#if 0
            blz::DV<FT, blz::rowVector> tmp(nd);
            blz::DM<FT> tmpmat(blz::rows(mat, asn.data(), asn.size()));
            coresets::l1_median(tmpmat, tmp);
            if(tmp != centers[i]) {
                //for(size_t j = 0; j < centers[i].size(); ++j) if(centers[i][j]) std::fprintf(stderr, "%zu/%g\n", j, centers[i][j]);
                for(size_t j = 0; j < nd; ++j) {
                    if((tmp[j] || centers[i][j]) && tmp[j] != centers[i][j]) {
                        std::fprintf(stderr, "at %zu found %g but expected %g, difference %g\n", j, centers[i][j], tmp[j], centers[i][j] - tmp[j]);
                    }
                }
            }
            auto dist = blz::l1Norm(tmp - centers[i]);
            assert(dist == 0 || !std::fprintf(stderr, "dist between sparse l1 and dnese l1 is %g (compared to %g)\n", dist, blz::l1Norm(centers[i])) || !&(std::cerr << (tmp - centers[i])));
#endif
            auto med1tend = std::chrono::high_resolution_clock::now();
            std::fprintf(stderr, "Median took %gms\n", util::timediff2ms(med1ts, med1tend));
        }
        // Set centers
        if(sa.size()) {
            std::fprintf(stderr, "reseeding %zu centers\n", sa.size());
            blz::DV<FT> probs(mat.rows());
            FT *pd = probs.data(), *pe = pd + probs.size();
            std::fill(asn.begin(), asn.end(), 0);
            std::vector<uint32_t> current_indices;
            for(unsigned i = 0; i < opts.k; ++i)
                if(std::find(sa.begin(), sa.end(), i) == sa.end()) current_indices.push_back(i);
            for(auto &i: asns) i.clear();
            std::fill(costs.begin(), costs.end(), std::numeric_limits<FT>::max());
            {
                auto check_idx = [&centers,&mat,ap=asn.data(),cp=costs.data()](auto idx, auto cid) {
                    if(const auto v = blz::l1Norm(centers[cid] - row(mat, idx, blz::unchecked)); v < cp[idx]) {
                        cp[idx] = v;
                        ap[idx] = cid;
                    }
                };
                for(size_t i = 0; i < costs.size(); ++i)
                    for(const auto cid: current_indices)
                        check_idx(i, cid);
            }
            for(const auto idx: sa) {
                std::fprintf(stderr, "idx being reseeded: %d\n", idx);
                //const auto idx = sa[i];
                std::partial_sum(costs.begin(), costs.end(), pd);
                std::uniform_real_distribution<double> dist;
                std::ptrdiff_t found = std::lower_bound(pd, pe, dist(rng) * pe[-1]) - pd;
                centers[idx] = row(mat, found);
                for(size_t i = 0; i < mat.rows(); ++i) {
                    auto c = blz::l1Norm(centers[idx] - row(mat, i));
                    if(c < costs[i]) {
                        asn[i] = idx;
                        costs[i] = c;
                    }
                }
            }
            for(unsigned i = 0; i < sa.size(); ++i)
                asns[asn[i]].push_back(i);
            sa.clear();
            goto center_setup;
        }
        std::fill(asn.begin(), asn.end(), 0);
        OMP_PFOR
        for(size_t i = 0; i < mat.rows(); ++i) {
            auto lhr = row(mat, i, blaze::unchecked);
            costs[i] = blz::l1Norm(lhr - centers[0]);
            for(unsigned j = 1; j < opts.k; ++j)
                if(auto v = blz::l1Norm(lhr - centers[j]);
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
get_ms_centers_l1(blz::SM<FT> &mat, unsigned k, [[maybe_unused]] unsigned maxiter, double eps, uint64_t seed)
{
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
            dm(j, i) = r[j] = blz::l1Norm(dr - row(mat, j, blz::unchecked));
        }
#ifndef NDEBUG
        if(unlikely(i % 16 == 0)) {
            std::fprintf(stderr, "Computed %zu/%zu rows in %gms\n", i, mat.rows(), util::timediff2ms(start, std::chrono::high_resolution_clock::now()));
        }
#endif
    }
    [[maybe_unused]] auto distmattime = std::chrono::high_resolution_clock::now();
    // Run JV
#if USE_JV_TOO
    std::fprintf(stderr, "[get_ms_centers_l1:%s:%d] Time to compute distance matrix: %gms\n", __FILE__, __LINE__, util::timediff2ms(start, distmattime));
    auto solver = jv::make_jv_solver(dm);
    auto [fac, asn] = solver.kmedian(k, maxiter);
    std::fprintf(stderr, "initial centers: "); for(const auto f: fac) std::fprintf(stderr, ",%d", f); std::fputc('\n', stderr);
    shared::sort(fac.begin(), fac.end());
#else
    std::vector<uint32_t> fac;
#endif
    // Local search from that solution
    auto lsearcher = make_kmed_lsearcher(dm, k, eps, seed);
    lsearcher.lazy_eval_ = 2;
#if USE_JV_TOO
    lsearcher.assign_centers(fac.begin(), fac.end());
#endif
    lsearcher.run();
    fac.assign(lsearcher.sol_.begin(), lsearcher.sol_.end());
    const auto fbeg = fac.data(), fend = &fac[k];
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

} // namespace clustering

using clustering::l1_sum_core;

} // namespace minicore
#endif
