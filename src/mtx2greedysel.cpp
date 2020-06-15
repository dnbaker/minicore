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
    std::mt19937_64 mt(opts.seed);
    auto centers = coresets::kcenter_greedy_2approx(app, app.size(), opts.k, mt);
    std::FILE *ofp;
    if(!(ofp = std::fopen((out + ".centers").data(), "w"))) throw 1;
    for(size_t i = 0; i < opts.k; ++i) {
        std::fprintf(ofp, "%u\n", centers[i]);
    }
    std::fclose(ofp);
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
