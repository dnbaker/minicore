#include "include/minicore/clustering/solve.h"
#include "blaze/util/Serialization.h"
#include "src/tests/solvetestdata.cpp"

namespace clust = minicore::clustering;
using namespace minicore;

// #define double float

int main(int argc, char *argv[]) {
    int NUMITER = 100;
    if(const char *s = std::getenv("NUMITER")) {
        NUMITER = std::atoi(s) > 0 ? std::atoi(s): 1;
    }
    const bool with_replacement = std::getenv("WITHREPS");
    //const bool use_importance_sampling = std::getenv("USE_IMPORTANCE_SAMPLING");
    std::srand(0);
    std::ios_base::sync_with_stdio(false);
    int nthreads = 1;
    unsigned int k = 10;
    //double temp = 1.;
    dist::DissimilarityMeasure msr = dist::MKL;
    blz::DV<double> prior{double(1)};
    for(int c;(c = getopt(argc, argv, "z:T:m:p:P:k:h?")) >= 0;) {switch(c) {
        //case 'T': temp = std::atof(optarg); break;
        case 'm': msr = (dist::DissimilarityMeasure)std::atoi(optarg); break;
        case 'P': prior[0] = std::atof(optarg); break;
        case 'p': nthreads = std::atoi(optarg); break;
        case 'k': k = std::atoi(optarg); break;
        case 'z': {
            blaze::Archive<std::ifstream> arch(optarg);
            x = blaze::DynamicMatrix<double>();
            try {
                arch >> x;
            } catch(const std::runtime_error &ex) {                                                     
                std::fprintf(stderr, "Could not get x from arch using >>. Error msg: %s\n", ex.what()); 
                throw;
            } catch(const std::exception &ex) {
                std::fprintf(stderr, "unknown failure. msg: %s\n", ex.what());
                throw;
            }
            break;
        }
        case '?':
        case 'h':dist::print_measures();  
                std::fprintf(stderr, "Usage: %s <flags> \n-z: load blaze matrix from path\n-P: set prior (1.)\n-T set temp [1.]\n-p set num threads\n -m Set measure (MKL, 5)\n-k: set k [10]\n", *argv);
                return EXIT_FAILURE;
    }}
    OMP_ONLY(omp_set_num_threads(nthreads);)
    if(std::find_if(argv, argc + argv, [](auto x) {return std::strcmp(x, "-h") == 0;}) != argc + argv) {
        dist::print_measures();
        std::exit(1); 
    }
    const size_t nr = x.rows(), nc = x.columns();
    std::fprintf(stderr, "prior: %g\n", prior[0]);
    std::fprintf(stderr, "msr: %d/%s\n", (int)msr, dist::msr2str(msr));
    std::vector<blaze::CompressedVector<double, blaze::rowVector>> centers;
    std::vector<blaze::CompressedVector<double, blaze::rowVector>> ocenters;
    std::vector<int> ids{1018, 2624, 5481, 6006, 8972};
    while(ids.size() < k) {
        auto rid = std::rand() % x.rows();
        if(std::find(ids.begin(), ids.end(), rid) == ids.end())
            ids.emplace_back(rid);
    }
    for(const auto id: ids) centers.emplace_back(row(x, id));
    ocenters = centers;
    const double psum = prior[0] * nc;
    blz::DV<uint32_t> asn(nr);
    blz::DV<double> rowsums = blaze::sum<blz::rowwise>(x);
    blz::DV<double> centersums = blaze::generate(centers.size(), [&](auto x) {return blz::sum(centers[x]);});
    assert(rowsums.size() == x.rows());
    assert(centersums.size() == centers.size());
    blz::DM<double> complete_hardcosts = blaze::generate(nr, k, [&](auto r, auto col) {
        return cmp::msr_with_prior(msr, row(x, r, blz::unchecked), centers[col], prior, psum, rowsums[r], centersums[col]);
    });
    blz::DV<double> hardcosts = blaze::generate(nr, [&](auto id) {
        auto r = row(complete_hardcosts, id, blaze::unchecked);
        auto it = std::min_element(r.begin(), r.end());
        asn[id] = it - r.begin();
        return *it;
    });
    auto mnc = blz::min(hardcosts);
    std::fprintf(stderr, "Total cost: %g. max cost: %g. min cost: %g. mean cost:%g\n", blz::sum(hardcosts), blz::max(hardcosts), mnc, blz::mean(hardcosts));
    std::vector<uint32_t> counts(k);
    for(const auto v: asn) ++counts[v];
    for(unsigned i = 0; i < k; ++i) {
        std::fprintf(stderr, "Center %d with sum %g has %u supporting, with total cost of assigned points %g\n", i, blz::sum(centers[i]), counts[i],
                     blz::sum(blz::generate(nr, [&](auto id) { return asn[id] == i ? hardcosts[id]: 0.;})));
    }
    assert(min(asn) == 0);
    assert(max(asn) == centers.size() - 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    clust::perform_hard_clustering(x, msr, prior, centers, asn, hardcosts);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "Wall time for clustering: %gms\n", std::chrono::duration<double, std::milli>(t2 - t1).count());
    std::fprintf(stderr, "Now performing minibatch clustering\n");
    size_t mbsize = 500;
    if(char *s = std::getenv("MBSIZE")) {
        mbsize = std::strtoull(s, nullptr, 10);
    }
    std::fprintf(stderr, "mbsize: %zu\n", mbsize);
    auto mbcenters = ocenters;
    std::vector<double> weights(x.rows(), 1.);
    std::fprintf(stderr, "minibatch clustering with no weights, %s replacement, %s importance sampling\n",  with_replacement ? "with": "without", "without");
    clust::perform_hard_minibatch_clustering(x, msr, prior, mbcenters, asn, hardcosts, (double *)nullptr, mbsize, NUMITER, 10, /*reseed_after=*/10, /*with_replacement=*/with_replacement, /*seed=*/1);
    auto mbuwcenters = ocenters;
    std::fprintf(stderr, "minibatch clustering with uniform weights, %s replacement, %s importance sampling\n",  with_replacement ? "with": "without", "without");
    clust::perform_hard_minibatch_clustering(x, msr, prior, mbuwcenters, asn, hardcosts,  weights.data(), mbsize, NUMITER, 10, /*reseed_after=*/10, /*with_replacement=*/with_replacement, /*seed=*/1);
#if 0
                                       const WeightT *weights=static_cast<WeightT *>(nullptr),
                                       size_t mbsize=1000,
                                       size_t maxiter=10000,
                                       size_t calc_cost_freq=100,
                                       bool with_replacement=true,
                                       int maxinrow=5,
                                       uint64_t seed=0)
#endif

    auto is_mbcenters = ocenters;
    std::fprintf(stderr, "minibatch clustering with uniform weights, %sreplacement, %s importance sampling\n", with_replacement ? "with": "without", "without");
    clust::perform_hard_minibatch_clustering(x, msr, prior, is_mbcenters, asn, hardcosts,  weights.data(), mbsize, NUMITER, 10, /*reseed_after=*/10, /*with_replacement=*/with_replacement, /*seed=*/1);
}
