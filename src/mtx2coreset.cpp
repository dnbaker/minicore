#include "minocore/minocore.h"

using namespace minocore;

namespace dist = blz::distance;

struct Opts {
    size_t kmc2_rounds = 0;
    bool load_csr = false;
    dist::DissimilarityMeasure dis = dist::JSD;
    dist::Prior prior = dist::DIRICHLET;
    double gamma = 1.;
    unsigned k = 10;
    size_t coreset_size = 1000;
    uint64_t seed = 0;
    unsigned extra_sample_tries = 5;
    unsigned lloyd_max_rounds = 1000;
    coresets::SensitivityMethod sm = coresets::BFL;
    // If nonzero, performs KMC2 with m kmc2_rounds as chain length
    // Otherwise, performs standard D2 sampling
};

Opts opts;

void usage() {
    std::fprintf(stderr, "mtx2coreset <flags> [input file=""] [output_dir=mtx2coreset_output]\n"
                         "-f: Use floats (instead of doubles)\n"
                         "-C: load csr format (4 files) rather than matrix.mtx\n"
                         ">>> Dissimilarity Measures<<<\n"
                         "-1: Use L1 Norm \n"
                         "-2: Use L2 Norm \n"
                         "-S: Use squared L2 Norm (k-means)\n"
                         "-M: Use multinomial KL divergence\n"
                         "-T: Use total variation distance\n"
                         ">>> Prior settings <<<\n"
                         "-N: Use no prior. Default: Dirichlet\n"
                         "-g: Use the Gamma/Beta prior and set gamma's value [default: 1.]\n"
                         ">>> Coreset Construction >>>\n"
                         "-c: Set coreset size [1000]\n"
                         "-k: k (number of clusters)\n"
                         "-K: Use KMC2 for D2 sampling rather than kmeans++. May be significantly faster, but may provide lower quality solution.\n"
                         "-h: Emit usage\n");
    std::exit(1);
}

template<typename FT, typename RNG>
auto get_initial_centers(minocore::DissimilarityApplicator<blz::SM<FT>> &app, RNG &rng,
                         Opts opts) {
    std::vector<uint32_t> indices, asn;
    blz::DV<FT> costs(app.size());
    if(opts.kmc2_rounds) {
        indices = coresets::kmc2(app, rng, app.size(), opts.k, opts.kmc2_rounds);
        auto [oasn, ocosts] = coresets::get_oracle_costs(app, app.size(), indices);
        costs = std::move(ocosts);
        asn.assign(oasn.data(), oasn.data() + oasn.size());
    } else {
        std::vector<FT> fcosts;
        std::tie(indices, asn, fcosts) = make_kmeanspp(app, opts.k, rng());
        //indices = std::move(initcenters);
        std::copy(fcosts.data(), fcosts.data() + fcosts.size(), costs.data());
    }
    return std::make_tuple(indices, asn, costs);
}

template<typename FT>
int m2ccore(std::string in, std::string out, Opts opts)
{
    blz::SM<FT> sm(opts.load_csr ? csc2sparse<FT>(in): mtx2sparse<FT>(in));
    std::fprintf(stderr, "Loaded matrix [%zu/%zu] via %s\n", sm.rows(), sm.columns(), opts.load_csr ? "CSR": "MTX");
    std::string cmd = "mkdir " + out;
    if(int i = std::system(cmd.data())) std::fprintf(stderr, "rc: %d\n", i);
    blz::DV<FT, blaze::rowVector> pc(1);
    pc[0] = opts.prior == dist::DIRICHLET ? FT(1): opts.gamma;
    auto ptr(&pc);
    if(opts.prior == dist::NONE) ptr = nullptr;
    minocore::DissimilarityApplicator<blz::SM<FT>> app(sm, opts.dis, opts.prior, ptr);
    wy::WyRand<uint64_t, 2> rng(opts.seed);
    coresets::CoresetSampler<FT, uint32_t> cs;
    std::vector<blz::DV<FT, blz::rowVector>> centers(opts.k);
    if(dist::detail::satisfies_d2(opts.dis)) {
        auto [indices, asn, costs] = get_initial_centers(app, rng, opts);
        auto total_cost = blz::sum(costs);
        std::fprintf(stderr, "Initial approx cost is %g\n", total_cost);
        if(opts.extra_sample_tries) {
            for(unsigned i = opts.extra_sample_tries; i--;) {
                auto [indices2, asn2, costs2] = get_initial_centers(app, rng, opts);
                auto ntotal_cost = blz::sum(costs2);
                if(ntotal_cost < total_cost) {
                    std::swap(indices, indices2); std::swap(asn, asn2); std::swap(costs, costs2);
                    std::fprintf(stderr, "Swapping from cost of %g to %g\n", total_cost, ntotal_cost);
                    total_cost = ntotal_cost;
                }
            }
        }
        for(unsigned i = 0; i < opts.k; ++i)
            centers[i] = row(app.data(), indices[i], blaze::unchecked);
        if(opts.lloyd_max_rounds > 0) {
            if(clustering::perform_lloyd_loop<clustering::HARD>(
                centers, asn, app, opts.k, costs, rng(), static_cast<const FT *>(nullptr),
                opts.lloyd_max_rounds, 1e-10
            ) != clustering::FINISHED)
            {
                std::fprintf(stderr, "Warning: Lloyd's loop reached maximum iterations [%u]\n", opts.lloyd_max_rounds);
            }
        }
        // At this point, we can do some initial clustering.
        // TODO: do several rounds of Lloyd's algorithm
        cs.make_sampler(app.size(), opts.k, costs.data(), asn.data(), nullptr, rng(), opts.sm);
    } else {
        throw NotImplementedError("Needed: non-d2 coresets");
    }
    return 0;
}

int main(int argc, char **argv) {
    std::string inpath, outpath;
    bool use_double = true;
    for(int c;(c = getopt(argc, argv, "s:c:k:g:KSMT12NCfh?")) >= 0;) {
        switch(c) {
            case 'h': case '?': usage(); break;
            case 'f': use_double = false; break;
            case 'c': opts.coreset_size = std::strtoull(optarg, nullptr, 10); break;
            case 'C': opts.load_csr = true; break;
            case 'g': opts.gamma = std::atof(optarg); opts.prior = dist::GAMMA_BETA; break;
            case 'k': opts.k = std::atoi(optarg); break;
            case '1': opts.dis = dist::L1; break;
            case '2': opts.dis = dist::L2; break;
            case 'S': opts.dis = dist::SQRL2; break;
            case 'T': opts.dis = dist::TVD; break;
            case 'M': opts.dis = dist::MKL; break;
            case 'K': opts.kmc2_rounds = std::strtoull(optarg, 0, 10); break;
            case 's': opts.seed = std::strtoull(optarg,0,10); break;
            case 'N': opts.prior = dist::NONE;
        }
    }
    if(argc == optind) usage();
    if(argc - 1 >= optind) {
        inpath = argv[optind];
        if(argc - 2 >= optind)
            outpath = argv[optind + 1];
    }
    if(outpath.empty()) {
        outpath = "mtx2coreset_output";
    }
    return use_double ? m2ccore<double>(inpath, outpath, opts)
                      : m2ccore<float>(inpath, outpath, opts);
}
