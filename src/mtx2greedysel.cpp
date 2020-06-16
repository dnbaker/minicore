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
    std::fprintf(stderr, "mtx2greedysel <flags> [input file=""] [output_dir=mtx2greedysel_output]\n"
                         "=== General/Formatting ===\n"
                         "-f: Use floats (instead of doubles)\n"
                         "-p: Set number of threads [1]\n"
                         "-x: Transpose matrix (to swap feature/instance labels) during loading.\n"
                         "-C: load csr format (4 files) rather than matrix.mtx\n\n\n"
                         "=== Dissimilarity Measures ===\n"
                         "-1: Use L1 Norm \n"
                         "-2: Use L2 Norm \n"
                         "-S: Use squared L2 Norm (k-means)\n"
                         "\n"
                         "-M: Use multinomial KL divergence\n"
                         "-H: Use Hellinger distance\n"
                         "-j: Use multinomial Jensen-Shannon divergence\n"
                         "-J: Use multinomial Jensen-Shannon metric (square root of JSD)\n"
                         "-P: Use probability squared L2 norm\n"
                         "-Q: Use probability L2 norm\n"
                         "-T: Use total variation distance\n"
                         "-b: Use Bhattacharya Metric\n"
                         "-Y: Use Bhattacharya Distance\n"
                         "-i: Use Itakura-Saito Distance [prior required]\n"
                         "-I: Use reverse Itakura-Saito Distance [prior required]\n"
                         "\n\n"
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



int main(int argc, char **argv) {
    opts.stamper_.reset(new util::TimeStamper("argparse"));
    std::string inpath, outpath;
    [[maybe_unused]] bool use_double = true;
    for(int c;(c = getopt(argc, argv, "s:c:k:g:p:K:L:HPBdjJxSMT12NCDfh?")) >= 0;) {
        switch(c) {
            case 'h': case '?': usage();          break;
            case 'p': OMP_ONLY(omp_set_num_threads(std::atoi(optarg));)       break;
            case 'c': opts.coreset_samples = std::strtoull(optarg, nullptr, 10); break;
            case '1': opts.dis = dist::L1;        break;
            case '2': opts.dis = dist::L2;        break;
            case 'H': opts.dis = dist::HELLINGER; break;
            case 'i': opts.dis = dist::ITAKURA_SAITO; break;
            case 'I': opts.dis = dist::REVERSE_ITAKURA_SAITO; break;
            case 'J': opts.dis = dist::JSM;       break;
            case 'M': opts.dis = dist::MKL;       break;
            case 'P': opts.dis = dist::PSL2;      break;
            case 'Q': opts.dis = dist::PL2;      break;
            case 'S': opts.dis = dist::SQRL2;     break;
            case 'T': opts.dis = dist::TVD;       break;
            case 'Y': opts.dis = dist::BHATTACHARYYA_DISTANCE; break;
            case 'b': opts.dis = dist::BHATTACHARYYA_METRIC; break;
            case 'j': opts.dis = dist::JSD;       break;
            case 'D': opts.discrete_metric_search = true; break;
            case 'g': opts.gamma = std::atof(optarg); opts.prior = dist::GAMMA_BETA; break;
            case 'k': opts.k = std::atoi(optarg); break;
            case 'K': opts.kmc2_rounds = std::strtoull(optarg, nullptr, 10); break;
            case 'L': opts.lloyd_max_rounds = std::strtoull(optarg, nullptr, 10); break;
            case 'B': opts.load_blaze = true; opts.load_csr = false; break;
            case 'C': opts.load_csr = true;       break;
            case 'd': opts.prior = dist::DIRICHLET; break;
            case 's': opts.seed = std::strtoull(optarg,0,10); break;
            case 'x': opts.transpose_data = true; break;
            case 'f': use_double = false;         break;
        }
    }
    if(argc == optind) usage();
    if(argc - 1 >= optind) {
        inpath = argv[optind];
        if(argc - 2 >= optind)
            outpath = argv[optind + 1];
    }
    if(outpath.empty()) {
        outpath = "mtx2greedysel_output.";
        outpath += std::to_string(uint64_t(std::time(nullptr)));
    }
#ifndef NDEBUG
    return use_double ? m2gcore<double>(inpath, outpath, opts)
                      : m2gcore<float>(inpath, outpath, opts);
#else
    return m2gcore<double>(inpath, outpath, opts); // Reduce compilation time
#endif
}
