#include "minocore/minocore.h"
#include "blaze/util/Serialization.h"

using namespace minocore;

namespace dist = blz::distance;

using minocore::util::timediff2ms;
template<typename FT> using CType = blz::DV<FT, blz::rowVector>;

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

Opts opts;

void usage() {
    std::fprintf(stderr, "mtx2coreset <flags> [input file=""] [output_dir=mtx2coreset_output]\n"
                         "=== General/Formatting ===\n"
                         "-f: Use floats (instead of doubles)\n"
                         "-p: Set number of threads [1]\n"
                         "-x: Transpose matrix (to swap feature/instance labels) during loading.\n"
                         "-C: load csr format (4 files) rather than matrix.mtx\n\n\n"
                         "=== Dissimilarity Measures ===\n"
                         "-1: Use L1 Norm \n"
                         "-2: Use L2 Norm \n"
                         "-S: Use squared L2 Norm (k-means)\n"
                         "-M: Use multinomial KL divergence\n"
                         "-j: Use multinomial Jensen-Shannon divergence\n"
                         "-J: Use multinomial Jensen-Shannon metric (square root of JSD)\n"
                         "-P: Use probability squared L2 norm\n"
                         "-T: Use total variation distance\n\n\n"
                         "=== Prior settings ===\n"
                         "-N: Use no prior. Default: Dirichlet\n"
                         "-g: Use the Gamma/Beta prior and set gamma's value [default: 1.]\n\n\n"
                         "=== Coreset Construction ===\n"
                         "-c: Set coreset size [1000]\n"
                         "-k: k (number of clusters)\n"
                         "-K: Use KMC2 for D2 sampling rather than kmeans++. May be significantly faster, but may provide lower quality solution.\n\n\n"
                         "-h: Emit usage\n");
    std::exit(1);
}

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
        std::tie(indices, asn, fcosts) = coresets::kmeanspp(matrix, rng, opts.k, blz::sqrL2Norm());
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

template<typename FT>
auto kmeans_sum_core(blz::SM<FT> &mat, std::string out, Opts opts) {
    wy::WyRand<uint64_t, 2> rng(opts.seed);
    std::vector<uint32_t> indices, asn;
    blz::DV<FT, blz::rowVector> costs;
    std::tie(indices, asn, costs) = repeatedly_get_initial_centers(mat, rng, opts.k, opts.kmc2_rounds, opts.extra_sample_tries);
    std::vector<blz::DV<FT, blz::rowVector>> centers(opts.k);
    { // write selected initial points to file
        std::ofstream ofs(out + ".initial_points");
        for(size_t i = 0; i < indices.size(); ++i) {
            ofs << indices[i] << ',';
        }
        ofs << '\n';
    }
    OMP_PFOR
    for(unsigned i = 0; i < opts.k; ++i) {
        centers[i] = row(mat, indices[i]);
        std::fprintf(stderr, "Center %u initialized by index %u and has sum of %g\n", i, indices[i], blz::sum(centers[i]));
    }
    FT tcost = blz::sum(costs), firstcost = tcost;
    //auto centerscpy = centers;
    size_t iternum = 0;
    OMP_ONLY(std::unique_ptr<std::mutex[]> mutexes(new std::mutex[opts.k]);)
    std::unique_ptr<uint32_t[]> counts(new uint32_t[opts.k]);
    for(;;) {
        std::fprintf(stderr, "[Iter %zu] Cost: %g\n", iternum, tcost);
        // Set centers
        center_setup:
        for(auto &c: centers) c = 0.;
        std::fprintf(stderr, "Performing sums\n");
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


template<typename FT>
int m2ccore(std::string in, std::string out, Opts opts)
{
    std::fprintf(stderr, "[%s] Starting main\n", __PRETTY_FUNCTION__);
    auto tstart = std::chrono::high_resolution_clock::now();
    blz::SM<FT> sm;
    if(opts.load_csr) {
        std::fprintf(stderr, "Trying to load from csr\n");
        sm = csc2sparse<FT>(in);
    } else if(opts.load_blaze) {
        std::fprintf(stderr, "Trying to load from blaze\n");
        blaze::Archive<std::ifstream> arch(in);
        arch >> sm;
    } else {
        std::fprintf(stderr, "Trying to load from mtx\n");
        sm = mtx2sparse<FT>(in, opts.transpose_data);
    }
    std::tuple<std::vector<CType<FT>>, std::vector<uint32_t>, CType<FT>> hardresult;
    std::tuple<std::vector<CType<FT>>, blz::DM<FT>, CType<FT>> softresult;

    switch(opts.dis) {
        case dist::SQRL2: case dist::PSL2: {
            if(opts.dis == dist::PSL2) {
                for(auto r: blz::rowiterator(sm)) r /= blz::sum(r);
            }
            if(opts.soft) {
                throw NotImplementedError("SQRL2/PSL2 under soft clustering");
            } else {
                hardresult = kmeans_sum_core(sm, out, opts);
            }
            break;
        }
        default: {
            std::fprintf(stderr, "%d/%s not supported\n", (int)opts.dis, blz::detail::prob2desc(opts.dis));
            throw NotImplementedError("Not yet");
        }
    }
    if(opts.soft) {
        throw 1;
    } else {
        auto &[centers, asn, costs] = hardresult;
        coresets::CoresetSampler<FT, uint32_t> cs;
        cs.make_sampler(sm.rows(), opts.k, costs.data(), asn.data(), nullptr, opts.seed, opts.sm);
        cs.write(out + ".coreset_sampler");
        std::FILE *ofp;
        if(!(ofp = std::fopen((out + ".centers").data(), "w"))) throw 1;
        std::fprintf(ofp, "#Center\tFeatures\t...\t...\n");
        for(size_t i = 0; i < opts.k; ++i) {
            std::fprintf(ofp, "%zu\t", i + 1);
            const auto &c(centers[i]);
            for(size_t j = 0; j < c.size() - 1; ++j)
                std::fprintf(ofp, "%0.12g\t", c[j]);
            std::fprintf(ofp, "%0.12g\n", c[opts.k - 1]);
        }
        std::fclose(ofp);
        if(!(ofp = std::fopen((out + ".assignments").data(), "w"))) throw 1;
        for(size_t i = 0; i < asn.size(); ++i) {
            std::fprintf(ofp, "%zu\t%u\n", i, asn[i]);
        }
        std::fclose(ofp);
        std::string fmt = sizeof(FT) == 4 ? ".float32": ".double";
        if(!(ofp = std::fopen((out + fmt + ".importance").data(), "w"))) throw 1;
        if(std::fwrite(cs.probs_.get(), sizeof(FT), cs.size(), ofp) != cs.size()) throw 2;
        std::fclose(ofp);
    }
    auto tstop = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "Full program took %gms\n", util::timediff2ms(tstart, tstop));
    return 0;
}

int main(int argc, char **argv) {
    std::string inpath, outpath;
    bool use_double = true;
    for(int c;(c = getopt(argc, argv, "s:c:k:g:p:K:BPjJxSMT12NCfh?")) >= 0;) {
        switch(c) {
            case 'h': case '?': usage();          break;
            case 'B': opts.load_blaze = true; opts.load_csr = false; break;
            case 'f': use_double = false;         break;
            case 'c': opts.coreset_size = std::strtoull(optarg, nullptr, 10); break;
            case 'C': opts.load_csr = true;       break;
            case 'p': OMP_ONLY(omp_set_num_threads(std::atoi(optarg));)       break;
            case 'g': opts.gamma = std::atof(optarg); opts.prior = dist::GAMMA_BETA; break;
            case 'k': opts.k = std::atoi(optarg); break;
            case '1': opts.dis = dist::L1;        break;
            case '2': opts.dis = dist::L2;        break;
            case 'S': opts.dis = dist::SQRL2;     break;
            case 'T': opts.dis = dist::TVD;       break;
            case 'M': opts.dis = dist::MKL;       break;
            case 'j': opts.dis = dist::JSD;       break;
            case 'J': opts.dis = dist::JSM;       break;
            case 'P': opts.dis = dist::PSL2;      break;
            case 'K': opts.kmc2_rounds = std::strtoull(optarg, nullptr, 10); break;
            case 's': opts.seed = std::strtoull(optarg,0,10); break;
            case 'N': opts.prior = dist::NONE;    break;
            case 'x': opts.transpose_data = true; break;
        }
    }
    if(argc == optind) usage();
    if(argc - 1 >= optind) {
        inpath = argv[optind];
        if(argc - 2 >= optind)
            outpath = argv[optind + 1];
    }
    if(outpath.empty()) {
        outpath = "mtx2coreset_output.";
        outpath += std::to_string(uint64_t(std::time(nullptr)));
    }
    return use_double ? m2ccore<double>(inpath, outpath, opts)
                      : m2ccore<float>(inpath, outpath, opts);
}
