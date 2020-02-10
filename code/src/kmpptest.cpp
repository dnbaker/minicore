#include "kmeans.h"
#include "kcenter.h"
#include <new>
#include <chrono>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

#define t std::chrono::high_resolution_clock::now

#ifndef FLOAT_TYPE
#define FLOAT_TYPE float
#endif
using namespace fgc;
using namespace fgc::coresets;

template<typename Mat, typename RNG>
void test_kccs(Mat &mat, RNG &rng, size_t npoints, double eps) {
    auto matrowit = blz::rowiterator(mat);
    auto start = t();
    auto cs = outliers::kcenter_coreset(matrowit.begin(), matrowit.end(), rng, npoints, eps,
                /*mu=*/1);
    auto maxv = *std::max_element(cs.indices_.begin(), cs.indices_.end());
    std::fprintf(stderr, "max index: %u\n", unsigned(maxv));
    auto stop = t();
    std::fprintf(stderr, "kcenter coreset took %gs\n", double((stop - start).count()) / 1e9);
    start = t();
    auto csmat = index2matrix(cs, mat);
    stop = t();
    std::fprintf(stderr, "kcenter compacting to coreset took %gs\n", double((stop - start).count()) / 1e9);
}

int main(int argc, char *argv[]) {
    if(std::find_if(argv, argv + argc, [](auto x) {return std::strcmp(x, "-h") == 0 || std::strcmp(x, "--help") == 0;}) != argv + argc) {
        std::fprintf(stderr, "Usage: %s [n ? 100000] [k ? 50] [d ? 40] [eps ? 0.5]\n", argv[0]);
        return 0;
    }
#ifdef _OPENMP
    auto nt = std::thread::hardware_concurrency();
    if(auto env = std::getenv("OMP_NUM_THREADS"); env) {
        nt = std::atoi(env);
    }
    omp_set_num_threads(nt);
    std::fprintf(stderr, "%d threads used\n", nt);
#endif
    std::srand(0);
    size_t n = argc == 1 ? 1000000: std::atoi(argv[1]);
    size_t npoints = argc <= 2 ? 500: std::atoi(argv[2]);
    size_t nd = argc <= 3 ? 40: std::atoi(argv[3]);
    double eps = argc <= 4 ? 0.5: std::atof(argv[3]);
    auto ptr = static_cast<std::vector<FLOAT_TYPE> *>(std::malloc(n * sizeof(std::vector<FLOAT_TYPE>)));
    //std::unique_ptr<std::vector<FLOAT_TYPE>[]> stuff(n);
    wy::WyRand<uint32_t, 2> gen;
    OMP_PRAGMA("omp parallel for")
    for(auto i = 0u; i < n; ++i) {
        new (ptr + i) std::vector<FLOAT_TYPE>(40);
        //stuff[i] = std::vector<FLOAT_TYPE>(400);
        for(auto &e: ptr[i]) e = FLOAT_TYPE(std::rand()) / RAND_MAX;
    }
    std::fprintf(stderr, "generated\n");
    blaze::DynamicMatrix<FLOAT_TYPE> mat(n, nd);
    OMP_PRAGMA("omp parallel for")
    for(auto i = 0u; i < n; ++i) {
        auto r = row(mat, i);
        std::memcpy(&r[0], &ptr[i][0], sizeof(FLOAT_TYPE) * nd);
    }
    auto start = t();
    auto centers = kmeanspp(ptr, ptr + n, gen, npoints);
    auto stop = t();
    std::fprintf(stderr, "Time for kmeans++: %gs\n", double((stop - start).count()) / 1e9);
    // centers contains [centers, assignments, distances]
    start = t();
    auto kmc2_centers = kmc2(ptr, ptr + n, gen, npoints, 200);
    stop = t();
    std::fprintf(stderr, "Time for kmc^2: %gs\n", double((stop - start).count()) / 1e9);
    // then, a coreset can be constructed
    start = t();
    auto kc = kcenter_greedy_2approx(ptr, ptr + n, gen, npoints);
    stop = t();
    std::fprintf(stderr, "Time for kcenter_greedy_2approx: %gs\n", double((stop - start).count()) / 1e9);
    start = t();
    auto centers2 = kmeanspp(mat, gen, npoints, blz::L1Norm());
    stop = t();
    std::fprintf(stderr, "Time for kmeans++ on L1 norm on matrix: %gs\n", double((stop - start).count()) / 1e9);
    test_kccs(mat, gen, npoints, eps);
    //for(const auto v: centers) std::fprintf(stderr, "Woo: %u\n", v);
    start = t();
    auto kmppmcs = kmeans_matrix_coreset(mat, npoints, gen, npoints * 2);
    stop = t();
    std::fprintf(stderr, "Time for kmeans++ matrix coreset: %gs\n", double((stop - start).count()) / 1e9);
    std::destroy_n(ptr, n);
    std::free(ptr);
    blaze::DynamicMatrix<FLOAT_TYPE> sqmat(20, 20);
    randomize(sqmat);
    sqmat = map(sqmat, [](auto x) {return x * x + 1e-15;});
    assert(min(sqmat) > 0.);
    {
        auto greedy_metric = kcenter_greedy_2approx(rowiterator(sqmat).begin(), rowiterator(sqmat).end(),
                                                    gen, /*k=*/3, MatrixLookup{});
    }
}
