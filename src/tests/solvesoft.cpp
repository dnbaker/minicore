#include "include/minicore/clustering/solve.h"
#include "blaze/util/Serialization.h"
#include "src/tests/solvetestdata.cpp"

namespace clust = minicore::clustering;
using namespace minicore;

// #define double float
#ifndef FLOAT_TYPE
#define FLOAT_TYPE double
#endif
#ifndef OTHER_FLOAT_TYPE
#define OTHER_FLOAT_TYPE float
#endif

int main(int argc, char *argv[]) {
    std::srand(0);
    std::ios_base::sync_with_stdio(false);
    dist::print_measures();
    if(std::find_if(argv, argc + argv, [](auto x) {return std::strcmp(x, "-h") == 0;}) != argc + argv)
        std::exit(1);
    dist::DissimilarityMeasure msr = dist::MKL;
    bool skip_empty = false, transpose = true;
    FLOAT_TYPE temp = 1.;
    int nrows = 500, ncols = 250;
    int k = 10;
    if(char *s = std::getenv("NROWS")) nrows = std::atoi(s);
    if(char *s = std::getenv("NCOLS")) ncols = std::atoi(s);
    int nthreads = 1;
    bool loaded_blaze = false;
    blz::DV<FLOAT_TYPE> prior{1.};
    if(char *s = std::getenv("OMP_NUM_THREADS")) nthreads = std::atoi(s);
    for(int c;(c = getopt(argc, argv, "M:z:m:p:P:k:t:TEh?")) >= 0;) {switch(c) {
        case 't': temp = std::atof(optarg); break;
        case 'T': transpose = false; break;
        case 'm': msr = (dist::DissimilarityMeasure)std::atoi(optarg); break;
        case 'P': prior[0] = std::atof(optarg); break;
        case 'p': nthreads = std::atoi(optarg); break;
        case 'k': k = std::atoi(optarg); break;
        case 'E': skip_empty = true; break;
        case 'C': {
            x = minicore::util::csc2sparse<FLOAT_TYPE>(optarg, skip_empty); break;
        }
        case 'M': {
            x = minicore::util::mtx2sparse<FLOAT_TYPE>(optarg, transpose); break;
        }
        case 'z': {
            blaze::Archive<std::ifstream> arch(optarg);
            x = blaze::CompressedMatrix<FLOAT_TYPE>();
            try {
                arch >> x;
                std::fprintf(stderr, "Shape of loaded blaze matrix: %zu/%zu\n", x.rows(), x.columns());
            } catch(const std::runtime_error &ex) {
                blaze::CompressedMatrix<OTHER_FLOAT_TYPE> cm;
                try {
                arch >> cm;
                x = cm;
                } catch(...) {
                    std::fprintf(stderr, "Could not get x from arch using >>. Error msg: %s\n", ex.what());
                    throw;
                }
            } catch(const std::exception &ex) {
                std::fprintf(stderr, "unknown failure. msg: %s\n", ex.what());
                throw;
            }
            loaded_blaze = true;
            break;
        }
        case '?':
        case 'h':dist::print_measures();
                std::fprintf(stderr, "Usage: %s <flags> \n-z: load blaze matrix from path\n-P: set prior (1.)\n-T set temp [1.]\n-p set num threads\n-m Set measure (MKL, 5)\n-k: set k [10]\t-T transpose mtx file\t-M parse mtx file from argument\n", *argv);
                return EXIT_FAILURE;
    }}
    OMP_ONLY(omp_set_num_threads(nthreads);)
    const size_t nr = x.rows(), nc = x.columns();
    std::fprintf(stderr, "prior for soft clustering is %g\n", prior[0]);
    std::fprintf(stderr, "temp for soft clustering is %g\n", temp);
    std::fprintf(stderr, "msr: %d/%s\n", (int)msr, dist::msr2str(msr));
    std::vector<blaze::CompressedVector<FLOAT_TYPE, blaze::rowVector>> centers;
    std::vector<int> ids;
    while(ids.size() < (unsigned)k) {
        auto rid = std::rand() % x.rows();
        if(std::find(ids.begin(), ids.end(), rid) == ids.end())
            ids.emplace_back(rid);
    }
    for(const auto id: ids) centers.emplace_back(row(x, id));
    const FLOAT_TYPE psum = prior[0] * nc;
    std::fprintf(stderr, "psum: %g. pv: %g\n", psum, prior[0]);
    blz::DV<FLOAT_TYPE> rowsums = blaze::sum<blz::rowwise>(x);
    blz::DV<FLOAT_TYPE> centersums = blaze::generate(k, [&](auto x) {return blz::sum(centers[x]);});
    blz::DV<uint32_t> asn(nr);
    blz::DM<FLOAT_TYPE> complete_hardcosts = blaze::generate(nr, k, [&](auto row, auto col) {
        return cmp::msr_with_prior(msr, blaze::row(x, row), centers[col], prior, psum, rowsums[row], centersums[col]);
    });
    blz::DV<FLOAT_TYPE> hardcosts = blaze::generate(nr, [&](auto id) {
        auto r = row(complete_hardcosts, id, blaze::unchecked);
        auto it = std::min_element(r.begin(), r.end());
        asn[id] = it - r.begin();
        return *it;
    });
    auto mnc = blz::min(hardcosts);
    std::fprintf(stderr, "Total cost: %g. max cost: %g. min cost: %g. mean cost:%g\n", blz::sum(hardcosts), blz::max(hardcosts), mnc, blz::mean(hardcosts));
    std::vector<uint32_t> counts(k);
    for(const auto v: asn) ++counts[v];
    for(int i = 0; i < k; ++i) {
        std::fprintf(stderr, "Center %d with sum %g has %u supporting, with total cost of assigned points %g\n", i, blz::sum(centers[i]), counts[i],
                     blz::sum(blz::generate(nr, [&](auto id) { return asn[id] == i ? hardcosts[id]: 0.;})));
    }
    //assert(min(asn) == 0);
    //assert(max(asn) == centers.size() - 1);
    // generate a coreset
    // recalculate now
    std::cerr << "Calculate costs\n";
    complete_hardcosts = blaze::generate(nr, k, [&](auto r, auto col) {
        return cmp::msr_with_prior(msr, row(x, r), centers[col], prior, psum, rowsums[r], centersums[col]);
    });
    std::cerr << "Perform clustering\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    clust::perform_soft_clustering(x, msr, prior, centers, complete_hardcosts, temp);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "Wall time for clustering: %gms\n", std::chrono::duration<FLOAT_TYPE, std::milli>(t2 - t1).count());
    auto centerdistmat = evaluate(blaze::generate(k, k, [&centers](auto x, auto y) {return blz::l1Norm(blz::abs(centers[x] - centers[y]));}));
    std::cerr << centerdistmat << '\n';
#if 0
    for(const auto ctr: centers)
        std::cerr << ctr << '\n';
#endif
}
