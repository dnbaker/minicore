#include "minocore/minocore.h"
#include "blaze/util/Serialization.h"

using namespace minocore;

namespace dist = blz::distance;

using minocore::util::timediff2ms;

struct Opts {
    size_t kmc2_rounds = 0;
    bool load_csr = false, transpose_data = false, load_blaze = false;
    dist::DissimilarityMeasure dis = dist::JSD;
    dist::Prior prior = dist::DIRICHLET;
    double gamma = 1.;
    unsigned k = 10;
    size_t coreset_size = 1000;
    uint64_t seed = 0;
    unsigned extra_sample_tries = 0;
    unsigned lloyd_max_rounds = 1000;
    unsigned sampled_number_coresets = 100;
    coresets::SensitivityMethod sm = coresets::BFL;
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

template<typename FT, typename RNG>
auto get_initial_centers(minocore::DissimilarityApplicator<blz::SM<FT>> &app, RNG &rng,
                         Opts opts) {
    std::vector<uint32_t> indices, asn;
    blz::DV<FT> costs(app.size());
    if(opts.kmc2_rounds) {
        std::fprintf(stderr, "Performing kmc\n");
        indices = coresets::kmc2(app, rng, app.size(), opts.k, opts.kmc2_rounds);
        auto [oasn, ocosts] = coresets::get_oracle_costs(app, app.size(), indices);
        costs = std::move(ocosts);
        asn.assign(oasn.data(), oasn.data() + oasn.size());
    } else {
        std::fprintf(stderr, "Performing kmeanspp\n");
        std::vector<FT> fcosts;
        std::tie(indices, asn, fcosts) = make_kmeanspp(app, opts.k, rng());
        //indices = std::move(initcenters);
        std::copy(fcosts.data(), fcosts.data() + fcosts.size(), costs.data());
    }
    assert(*std::max_element(indices.begin(), indices.end()) < app.size());
    return std::make_tuple(indices, asn, costs);
}

template<typename FT>
int m2ccore(std::string in, std::string out, Opts opts)
{
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
#if 0
    //std::ofstream tmp("plain.txt");
    //tmp << sm;
    //blz::DV<FT, blz::rowVector> means(blz::mean<blz::columnwise>(sm));
    std::fprintf(stderr, "means: ");
    for(size_t i = 0; i < means.size(); ++i) {
        std::fprintf(stderr, "%zu/%g\n", i, means[i]);
    }
#endif
    if(opts.load_csr && opts.transpose_data) sm.transpose();
    std::fprintf(stderr, "Loaded matrix [%zu/%zu] via %s%s in %gms\n", sm.rows(), sm.columns(), opts.load_csr ? "CSR": "MTX", opts.transpose_data ? "(transposed)": "",
                 timediff2ms(tstart, std::chrono::high_resolution_clock::now()));
    std::string cmd = "mkdir -p " + out;
    if(int i = std::system(cmd.data())) std::fprintf(stderr, "rc: %d\n", i);
    blz::DV<FT, blaze::rowVector> pc(1);
    pc[0] = opts.prior == dist::DIRICHLET ? FT(1): opts.gamma;
    auto ptr(&pc);
    if(opts.prior == dist::NONE) ptr = nullptr;
    minocore::DissimilarityApplicator<blz::SM<FT>> app(sm, opts.dis, opts.prior, ptr);
    wy::WyRand<uint64_t, 2> rng(opts.seed);
    coresets::CoresetSampler<FT, uint32_t> cs;
    std::vector<blz::DV<FT, blz::rowVector>> centers(opts.k);
    std::vector<uint32_t> indices, asn;
    blz::DV<FT> costs;
    if(dist::use_scaled_centers(app.get_measure())) {
        std::fprintf(stderr, "Weighted centers\n");
        blz::DV<FT, blz::rowVector> r(app.weighted_row(0));
        blz::DV<FT, blz::rowVector> r1(app.weighted_row(1));
        blz::DV<FT, blz::rowVector> rnw(app.row(0));
        std::fprintf(stderr, "by idx with 0: %g. by value with copied vector: %g. by value, unweighted %g\n", app(1, 0), app(1, r), app(1, rnw));
        std::fprintf(stderr, "True distance: %g\n", sqrNorm(r - r1));
        assert(app(1, 0) == app(1, r));
    }
    if(dist::detail::satisfies_d2(opts.dis)) {
        auto approxstart = std::chrono::high_resolution_clock::now();
        std::tie(indices, asn, costs) = get_initial_centers(app, rng, opts);
        auto total_cost = blz::sum(costs);
        std::fprintf(stderr, "Initial approx cost is %0.12g\n", total_cost);
        if(opts.extra_sample_tries) {
            for(unsigned i = opts.extra_sample_tries; i--;) {
                auto [indices2, asn2, costs2] = get_initial_centers(app, rng, opts);
                auto ntotal_cost = blz::sum(costs2);
                if(ntotal_cost < total_cost) {
                    std::swap(indices, indices2); std::swap(asn, asn2); std::swap(costs, costs2);
                    std::fprintf(stderr, "Swapping from cost of %0.12g to %0.12g\n", total_cost, ntotal_cost);
                    total_cost = ntotal_cost;
                }
            }
        }
        for(unsigned i = 0; i < opts.k; ++i) {
            auto idx = indices[i];
            if(dist::use_scaled_centers(app.get_measure())) {
                centers[i] = app.weighted_row(idx);
            } else {
                std::fprintf(stderr, "Unweighted centers\n");
                centers[i] = app.row(idx);
            }
        }
        std::fprintf(stderr, "Set centers\n");
#ifndef NDEBUG
        std::vector<uint32_t> manual_asn(costs.size());
        for(size_t i = 0; i < app.size(); ++i) {
            auto cost = app(i, indices[0]);
            assert(cost == app(i, centers[0]) || !std::fprintf(stderr, "cost1: %g, 2: %g\n", cost, app(i, centers[0])));
            unsigned v = 0;
            for(unsigned j = 1; j < opts.k; ++j) {
                auto newcost = app(i, indices[j]);
                if(newcost < cost) v = j, cost = newcost;
                assert(app(i, indices[j]) == app(i, centers[j]));
            }
            manual_asn[i] = v;
        }
        std::vector<uint32_t> counts(indices.size());
        for(size_t i = 0; i < asn.size(); ++i) {
            assert(manual_asn[i] == asn[i]);
            if(manual_asn[i] != asn[i]) {
                std::fprintf(stderr, "mismatch: manual %u, asn %u. Costs; %g, %g\n", manual_asn[i], asn[i], app(i, indices[asn[i]]), app(i, indices[manual_asn[i]]));
            }
        }
        for(size_t i = 0; i < asn.size(); ++i)
            ++counts[asn[i]];
        for(size_t i = 0; i < counts.size(); ++i) {
            std::fprintf(stderr, "After initial seeding, assignments are: %zu/%u\n", i, counts[i]);
        }
#endif
        auto approxstop = std::chrono::high_resolution_clock::now();
        std::fprintf(stderr, "Approximate solution took %gms\n", timediff2ms(approxstart, approxstop));
        if(opts.lloyd_max_rounds > 0) {
            if(clustering::perform_lloyd_loop<clustering::HARD>(
                centers, asn, app, opts.k, costs, rng(), static_cast<const FT *>(nullptr),
                opts.lloyd_max_rounds, 1e-10
            ) != clustering::FINISHED)
            {
                std::fprintf(stderr, "Warning: Lloyd's loop reached maximum iterations [%u]\n", opts.lloyd_max_rounds);
            }
        }
        auto lloydstop = std::chrono::high_resolution_clock::now();
        total_cost = blz::sum(costs);
        std::fprintf(stderr, "Lloyd search took %gms and has total cost %g\n", timediff2ms(lloydstop, approxstop), total_cost);

        // At this point, we can do some initial clustering.
        // TODO: do several rounds of Lloyd's algorithm
        cs.make_sampler(app.size(), opts.k, costs.data(), asn.data(), nullptr, rng(), opts.sm);
    } else {
        throw NotImplementedError("Needed: non-d2 coresets");
    }
    cs.write(out + "/sampler.css");
    { // Sampled points
        std::FILE *ofp, *rfp;
        if((ofp = std::fopen((out + "/selected_points.tsv").data(), "wb")) == nullptr) throw 1;
        if((rfp = std::fopen((out + "/selected_points.bin").data(), "wb")) == nullptr) throw 2;
        for(unsigned i = 0; i < opts.sampled_number_coresets; ++i) {
            auto sampled = cs.sample(opts.coreset_size);
            sampled.write(rfp);
            for(size_t i = 0, e = sampled.size(); i < e; ++i) {
                std::fprintf(ofp, "%u\t%0.16g\n", sampled.indices_[i], sampled.weights_[i]);
            }
            std::fprintf(ofp, "===\n");
        }
        std::fclose(ofp); std::fclose(rfp);
    }

    { // Clustering result
        std::ofstream ofs(out + "/clustering.out");
        for(const auto &i: centers) ofs << i;

        std::FILE *fp = std::fopen((out + "/clustering.tsv").data(), "a+");
        if(!fp) throw 1;
        std::fprintf(fp, "ID\tCluster Assignment\tCost\tSampling probability\n");
        for(size_t i = 0; i < app.size(); ++i)
            std::fprintf(fp, "%zu\t%u\t%0.12g\t%0.12g\n", i, asn[i], costs[i], cs.probs_[i]);
        std::fclose(fp);
    }
    auto tstop = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "Full program took %gms\n", timediff2ms(tstart, tstop));
    // Save results
    return 0;
}

int main(int argc, char **argv) {
    std::string inpath, outpath;
    bool use_double = true;
    for(int c;(c = getopt(argc, argv, "s:c:k:g:p:BPjJxKSMT12NCfh?")) >= 0;) {
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
            case 'K': opts.kmc2_rounds = std::strtoull(optarg, 0, 10); break;
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
        outpath = "mtx2coreset_output";
    }
    return use_double ? m2ccore<double>(inpath, outpath, opts)
                      : m2ccore<float>(inpath, outpath, opts);
}
