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
                         "=== Mode settings ===\n"
                         "Default: cluster\n"
                         "-G: use greedy farthest-point selection (k-center 2-approximation)\n"
                         "-l: use D2 sampling\n"
                         "-7: use k-center coreset for clustering in doubling metrics\n"
                         "-O: outlier fraction to use for k-center clustering with outliers -G. Implies -G\n"
                        "\n\n\n"
                         "=== General/Formatting ===\n"
                         "-f: Use floats (instead of doubles)\n"
                         "-p: Set number of threads [1]\n"
                         "-x: Transpose matrix (to swap feature/instance labels) during loading.\n"
                         "-s: Set random seed\n"
                         "-C: load csr format (4 files) rather than matrix.mtx\n"
                         "-B: load .blaze/.blz format rather than .mtx\n\n\n"
                         "=== Dissimilarity Measures ===\n"
                         "-1: Use L1 Norm \n"
                         "-2: Use L2 Norm \n"
                         "-S: Use squared L2 Norm (k-means)\n"
                         "-M: Use multinomial KL divergence\n"
                         "-R: Use reverse multinomial KL divergence\n"
                         "-j: Use multinomial Jensen-Shannon divergence\n"
                         "-J: Use multinomial Jensen-Shannon metric (square root of JSD)\n"
                         "-P: Use probability squared L2 norm\n"
                         "-Q: Use probability L2 norm\n"
                         "-T: Use total variation distance\n"
                         "-i: Use Itakura-Saito distance\n"
                         "-I: Use reverse Itakura-Saito distance\n"
                         "-H: Use Hellinger distance\n"
                         "-u: Use Probability Cosine distance\n"
                         "-U: Use Cosine distance\n"
                        "\n\n\n"
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
int m2d2core(std::string in, std::string out, Opts &opts)
{
    auto &ts = *opts.stamper_;
    std::fprintf(stderr, "[%s] Starting main\n", __PRETTY_FUNCTION__);
    std::fprintf(stderr, "Parameters: %s\n", opts.to_string().data());
    ts.add_event("Parse matrix");
    blz::SM<FT> sm;
    opts.stamper_->add_event("load matrix");
    if(opts.load_csr) {
        std::fprintf(stderr, "Trying to load from csr\n");
        sm = csc2sparse<FT>(in);
    } else if(opts.load_blaze) {
        std::fprintf(stderr, "Trying to load from blaze %s\n", in.data());
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


#if 0
template<typename Iter, typename FT=shared::ContainedTypeFromIterator<Iter>,
         typename IT=std::uint32_t, typename RNG, typename Norm=L2Norm>
coresets::IndexCoreset<IT, FT>
kcenter_coreset_outliers(Iter first, Iter end, RNG &rng, size_t k, double eps=0.1, double mu=.5,
                double rho=1.5,
                double gamma=0.001, double eta=0.01, const Norm &norm=Norm()) {
    // rho is 'D' for R^D (http://www.wisdom.weizmann.ac.il/~robi/teaching/2014b-SeminarGeometryAlgorithms/lecture1.pdf)
    // in Euclidean space, as worst-case, but usually better in real data with structure.
    assert(mu > 0. && mu <= 1.);
    const size_t np = end - first;
#endif

template<typename FT>
int m2kccs(std::string in, std::string out, Opts &opts)
{
    auto &ts = *opts.stamper_;
    std::fprintf(stderr, "[%s] Starting main\n", __PRETTY_FUNCTION__);
    std::fprintf(stderr, "Parameters: %s\n", opts.to_string().data());
    ts.add_event("Parse matrix");
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
    std::vector<uint32_t> centers;
    auto kccs = kcenter_coreset_outliers(app, app.size(), mt, opts.k, opts.eps, 0.1, 1.5, opts.outlier_fraction);
    std::FILE *ofp;
    if(!(ofp = std::fopen((out + ".centers").data(), "w"))) throw 1;
    for(size_t i = 0; i < kccs.size(); ++i) {
        std::fprintf(ofp, "%u\t%g\n", kccs.indices_[i], kccs.weights_[i]);
    }
    std::fclose(ofp);
    return 0;
}

template<typename FT>
int m2greedycore(std::string in, std::string out, Opts &opts)
{
    auto &ts = *opts.stamper_;
    std::fprintf(stderr, "[%s] Starting main\n", __PRETTY_FUNCTION__);
    std::fprintf(stderr, "Parameters: %s\n", opts.to_string().data());
    ts.add_event("Parse matrix");
    blz::SM<FT> sm;
    opts.stamper_->add_event("load matrix");
    if(opts.load_csr) {
        std::fprintf(stderr, "Trying to load from csr\n");
        sm = csc2sparse<FT>(in);
    } else if(opts.load_blaze) {
        std::fprintf(stderr, "Trying to load from blaze %s\n", in.data());
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
    std::vector<uint32_t> centers;
    if(opts.outlier_fraction) {
        centers = coresets::kcenter_greedy_2approx_outliers(
            app, app.size(), mt, opts.k,
            /*eps=*/1.5, opts.outlier_fraction
        );
    } else centers = coresets::kcenter_greedy_2approx(app, app.size(), opts.k, mt);
    std::FILE *ofp;
    if(!(ofp = std::fopen((out + ".centers").data(), "w"))) throw 1;
    for(size_t i = 0; i < opts.k; ++i) {
        std::fprintf(ofp, "%u\n", centers[i]);
    }
    std::fclose(ofp);
    return 0;
}


template<typename FT>
int m2ccore(std::string in, std::string out, Opts &opts)
{
    std::fprintf(stderr, "[%s] Starting main\n", __PRETTY_FUNCTION__);
    std::fprintf(stderr, "Parameters: %s\n", opts.to_string().data());
    auto &ts = *opts.stamper_;
    ts.add_event("Parse matrix");
    auto tstart = std::chrono::high_resolution_clock::now();
    blz::SM<FT> sm;
    if(opts.load_csr) {
        std::fprintf(stderr, "Trying to load from csr\n");
        sm = csc2sparse<FT>(in);
    } else if(opts.load_blaze) {
        std::fprintf(stderr, "Trying to load from blaze %s\n", in.data());
        blaze::Archive<std::ifstream> arch(in);
        arch >> sm;
    } else {
        std::fprintf(stderr, "Trying to load from mtx\n");
        sm = mtx2sparse<FT>(in, opts.transpose_data);
    }
    std::tuple<std::vector<CType<FT>>, std::vector<uint32_t>, CType<FT>> hardresult;
    std::tuple<std::vector<CType<FT>>, blz::DM<FT>, CType<FT>> softresult;

    ts.add_event("Initial solution");
    switch(opts.dis) {
        case dist::L1: case dist::TVD: {
            assert(min(sm) >= 0.);
            if(opts.dis == dist::TVD) for(auto r: blz::rowiterator(sm)) r /= blz::sum(r);
            if(opts.soft) {
                throw NotImplementedError("L1/TVD under soft clustering");
            } else {
                hardresult = l1_sum_core(sm, out, opts);
                std::fprintf(stderr, "Total cost: %g\n", blz::sum(std::get<2>(hardresult)));
            }
            break;
        }
        case dist::L2: case dist::PL2: {
            assert(min(sm) >= 0.);
            if(opts.dis == dist::PL2) for(auto r: blz::rowiterator(sm)) r /= blz::sum(r);
            if(opts.soft) {
                throw NotImplementedError("L2/PL2 under soft clustering");
            } else {
                hardresult = l2_sum_core(sm, out, opts);
                std::fprintf(stderr, "Total cost: %g\n", blz::sum(std::get<2>(hardresult)));
            }
            break;
        }
        case dist::SQRL2: case dist::PSL2: {
            if(opts.dis == dist::PSL2) for(auto r: blz::rowiterator(sm)) r /= blz::sum(r);
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
        ts.add_event("Build coreset sampler");
        coresets::CoresetSampler<FT, uint32_t> cs;
        cs.make_sampler(sm.rows(), opts.k, costs.data(), asn.data(), nullptr, opts.seed, opts.sm);
        ts.add_event("Write summary data to disk");
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
    }
    auto tstop = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "Full program took %gms\n", util::timediff2ms(tstart, tstop));
    return 0;
}

enum ResultType {
    CORESET,
    GREEDY_SELECTION,
    D2_SAMPLING,
    DOUBLING_METRIC_CORESET
};

int main(int argc, char **argv) {
    opts.stamper_.reset(new util::TimeStamper("argparse"));
    std::fprintf(stderr, "[CLI: '");
    for(char **av = argv; *av; std::fprintf(stderr, "%s ", *av++));
    std::fprintf(stderr, "']\n");
    std::string inpath, outpath;
    [[maybe_unused]] bool use_double = true;
    ResultType rt = ResultType::CORESET;
    for(int c;(c = getopt(argc, argv, "s:c:k:g:p:K:L:O:uURlGHiIYQbFVP7BdjJxSMT12NCDfh?")) >= 0;) {
        switch(c) {
            case 'p': OMP_ONLY(omp_set_num_threads(std::atoi(optarg));) break;
            case 'h': case '?': usage();          break;

            case '1': opts.dis = dist::L1;        break;
            case '2': opts.dis = dist::L2;        break;
            case 'H': opts.dis = dist::HELLINGER; break;
            case 'I': opts.dis = dist::REVERSE_ITAKURA_SAITO; break;
            case 'J': opts.dis = dist::JSM;       break;
            case 'M': opts.dis = dist::MKL;       break;
            case 'P': opts.dis = dist::PSL2;      break;
            case 'Q': opts.dis = dist::PL2;       break;
            case 'S': opts.dis = dist::SQRL2;     break;
            case 'T': opts.dis = dist::TVD;       break;
            case 'Y': opts.dis = dist::BHATTACHARYYA_DISTANCE; break;
            case 'b': opts.dis = dist::BHATTACHARYYA_METRIC; break;
            case 'i': opts.dis = dist::ITAKURA_SAITO; break;
            case 'j': opts.dis = dist::JSD;       break;
            case 'u': opts.dis = dist::PROBABILITY_COSINE_DISTANCE; break;
            case 'U': opts.dis = dist::COSINE_DISTANCE; break;
            case 'R': opts.dis = dist::REVERSE_MKL; break;

            case 'F': opts.sm = coresets::FL; break;
            case 'V': opts.sm = coresets::VARADARAJAN_XIAO; break;
            case 'E': opts.sm = coresets::LBK; break;

            case 'G': rt = ResultType::GREEDY_SELECTION; break;
            case '7': rt = ResultType::DOUBLING_METRIC_CORESET; break;
            case 'O': opts.outlier_fraction = std::atof(optarg); break;
			case 'l': rt = ResultType::D2_SAMPLING; break;

            case 'g': opts.gamma = std::atof(optarg); opts.prior = dist::GAMMA_BETA; break;
            case 'd': opts.prior = dist::DIRICHLET; break;

            case 'c': opts.coreset_samples = std::strtoull(optarg, nullptr, 10); break;
            case 'D': opts.discrete_metric_search = true; break;

            case 'k': opts.k = std::atoi(optarg); break;
            case 'K': opts.kmc2_rounds = std::strtoull(optarg, nullptr, 10); break;
            case 'L': opts.lloyd_max_rounds = std::strtoull(optarg, nullptr, 10); break;
            case 'B': opts.load_blaze = true; opts.load_csr = false; break;
            case 'C': opts.load_csr = true;       break;
            case 's': opts.seed = std::strtoull(optarg,0,10); break;
            case 'x': opts.transpose_data = true; break;
            case 'f': use_double = false;         break;
        }
    }
    if(dist::detail::is_bregman(opts.dis) && opts.sm != coresets::LBK && rt == CORESET) {
        std::fprintf(stderr, "Bregman divergences need LBK coreset construction. Switching to it from %s\n", coresets::sm2str(opts.sm));
        opts.sm = coresets::LBK;
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
    opts.stamper_->add_event("dispatch main");
    // Only compile version with doubles in debug mode to reduce compilation time
    switch(rt) {
#ifndef NDEBUG
        case GREEDY_SELECTION: {
            return use_double ? m2greedycore<double>(inpath, outpath, opts):
                                m2greedycore<float>(inpath, outpath, opts);
        }
        case D2_SAMPLING: {
            return use_double ? m2d2core<double>(inpath, outpath, opts)
                              : m2d2core<float>(inpath, outpath, opts);
        }
        case CORESET:
            return use_double ? m2ccore<double>(inpath, outpath, opts)
                              : m2ccore<float>(inpath, outpath, opts);
        case DOUBLING_METRIC_CORESET:
            return use_double ? m2kccs<double>(inpath, outpath, opts)
                              : m2kccs<float>(inpath, outpath, opts);
#else
	case CORESET: 	       return m2ccore<double>(inpath, outpath, opts);
	case GREEDY_SELECTION: return m2greedycore<double>(inpath, outpath, opts);
	case D2_SAMPLING:      return m2d2core<double>(inpath, outpath, opts);
#endif
	default: HEDLEY_UNREACHABLE();
    }
    return 1; // Never happens
}
