#include "minocore/minocore.h"
#include "blaze/util/Serialization.h"
#include "minocore/clustering/sqrl2.h"
#include "minocore/clustering/l2.h"
#include "minocore/clustering/l1.h"

using namespace minocore;

namespace dist = blz::distance;

using minocore::util::timediff2ms;

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
                         "-d: Use Dirichlet prior. Default: no prior.\n"
                         "-g: Use the Gamma/Beta prior and set gamma's value [default: 1.]\n\n\n"
                         "=== Optimizer settings ===\n"
                         "-D: Use metric solvers before EM rather than D2 sampling\n\n\n"
                         "=== Coreset Construction ===\n"
                         "-c: Set coreset size [1000]\n"
                         "-k: k (number of clusters)\n"
                         "-K: Use KMC2 for D2 sampling rather than kmeans++. May be significantly faster, but may provide lower quality solution.\n\n\n"
                         "-L: Use max [param] rounds in search. [1000]\n"
                         "-h: Emit usage\n");
    std::exit(1);
}


template<typename FT>
int m2ccore(std::string in, std::string out, Opts &opts)
{
    auto &ts = *opts.stamper_;
    std::fprintf(stderr, "[%s] Starting main\n", __PRETTY_FUNCTION__);
    std::fprintf(stderr, "Parameters: %s\n", opts.to_string().data());
    ts.add_event("Parse matrix");
    auto tstart = std::chrono::high_resolution_clock::now();
    blz::SM<FT> sm;
    opts.stamper_->add_event("load matrix");
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
    std::fprintf(stderr, "Loaded\n");

    blz::DV<FT, blz::rowVector> pc(1);
    blz::DV<FT, blz::rowVector> *pcp = nullptr;
    pcp = &pc;
    if(opts.prior == dist::DIRICHLET) pc[0] = 1.;
    else if(opts.prior == dist::GAMMA_BETA) pc[0] = opts.gamma;
    if(opts.prior != dist::NONE)
        pcp = &pc;
    ts.add_event("Set up applicator + caching");
    auto app = jsd::make_probdiv_applicator(sm, opts.dis, opts.prior, pcp);
    std::fprintf(stderr, "made applicator\n");
    ts.add_event("D^2 sampling");
    auto [centers, asn, costs] = jsd::make_kmeanspp(app, opts.k, opts.seed);
    auto csum = blz::sum(costs);
    for(unsigned i = 0; i < opts.extra_sample_tries; ++i) {
        auto [centers2, asn2, costs2] = jsd::make_kmeanspp(app, opts.k, opts.seed);
        auto csum2 = blz::sum(costs2);
        if(csum2 < csum) std::tie(centers, asn, costs, csum) = std::move(std::tie(centers2, asn2, costs2, csum2));
    }
    std::fprintf(stderr, "sampled points\n");
    
    ts.add_event("Coreset sampler");
    coresets::CoresetSampler<FT, uint32_t> cs;
    cs.make_sampler(sm.rows(), opts.k, costs.data(), asn.data(), nullptr, opts.seed, opts.sm, opts.fl_b, centers.data());
    ts.add_event("Save results");
    std::FILE *ofp;
    if(!(ofp = std::fopen((out + ".centers").data(), "w"))) throw 1;
    for(size_t i = 0; i < opts.k; ++i) {
        std::fprintf(ofp, "%u\n", centers[i]);
    }
    std::fclose(ofp);
    if(!(ofp = std::fopen((out + ".assignments").data(), "w"))) throw 1;
    for(size_t i = 0; i < asn.size(); ++i) {
        std::fprintf(ofp, "%zu\t%u\n", i, asn[i]);
    }
    std::fclose(ofp);
    std::string fmt = sizeof(FT) == 4 ? ".float32": ".double";
    if(!(ofp = std::fopen((out + fmt + ".importance").data(), "w"))) throw 1;
    if(std::fwrite(cs.probs_.get(), sizeof(cs.probs_[0]), cs.size(), ofp) != cs.size()) throw 2;
    std::fclose(ofp);
    if(!(ofp = std::fopen((out + fmt + ".costs").data(), "w"))) throw 1;
    if(std::fwrite(costs.data(), sizeof(FT), costs.size(), ofp) != costs.size()) throw 3;
    std::fclose(ofp);
    if(!(ofp = std::fopen((out + fmt + ".samples").data(), "w"))) throw 1;
    ts.add_event(std::string("Sample ") + std::to_string(opts.coreset_samples) + " points");
    std::unique_ptr<uint32_t[]> indices(new uint32_t[opts.coreset_samples]);
    cs.sample(&indices[0], &indices[opts.coreset_samples]);
    if(std::fwrite(indices.get(), sizeof(indices[0]), opts.coreset_samples, ofp) != opts.coreset_samples)
        throw std::runtime_error("3");
    std::fclose(ofp);
    if(!(ofp = std::fopen((out + fmt + ".samples.txt").data(), "w"))) throw 1;
    for(size_t i = 0; i < opts.coreset_samples; ++i)
        std::fprintf(ofp, "%u\t%0.12g\n", unsigned(indices[i]), cs.probs_[indices[i]]);
    std::fclose(ofp);
    if(opts.sm == coresets::FL) {
        for(const auto sz: {100, 1000, 10000, 100000}) {
            if(!(ofp = std::fopen((out + fmt + '.' + std::to_string(sz) + ".samples.txt").data(), "w"))) throw 1;
            auto cso = cs.sample(sz);
            cso.compact();
            for(size_t i = 0; i < cso.size(); ++i) std::fprintf(ofp, "%u\t%0.12g\n", cso.indices_[i], cso.weights_[i]);
            std::fclose(ofp);
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    opts.stamper_.reset(new util::TimeStamper("argparse"));
    std::string inpath, outpath;
    [[maybe_unused]] bool use_double = true;
    for(int c;(c = getopt(argc, argv, "s:c:k:g:p:K:L:PBdjJxSMT12NCDfh?")) >= 0;) {
        switch(c) {
            case 'h': case '?': usage();          break;
            case 'B': opts.load_blaze = true; opts.load_csr = false; break;
            case 'f': use_double = false;         break;
            case 'c': opts.coreset_samples = std::strtoull(optarg, nullptr, 10); break;
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
            case 'L': opts.lloyd_max_rounds = std::strtoull(optarg, nullptr, 10); break;
            case 'J': opts.dis = dist::JSM;       break;
            case 'P': opts.dis = dist::PSL2;      break;
            case 'K': opts.kmc2_rounds = std::strtoull(optarg, nullptr, 10); break;
            case 's': opts.seed = std::strtoull(optarg,0,10); break;
            case 'd': opts.prior = dist::DIRICHLET;    break;
            case 'D': opts.discrete_metric_search = true; break;
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
#ifndef NDEBUG
    return use_double ? m2ccore<double>(inpath, outpath, opts)
                      : m2ccore<float>(inpath, outpath, opts);
#else
    return m2ccore<double>(inpath, outpath, opts); // Reduce compilation time
#endif
}
