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

int main(int argc, char *argv[]) {
#ifdef _OPENMP
    auto nt = std::thread::hardware_concurrency();
    omp_set_num_threads(nt);
#endif
    std::srand(0);
    size_t n = argc == 1 ? 100000: std::atoi(argv[1]);
    size_t npoints = argc <= 2 ? 50: std::atoi(argv[2]);
    size_t nd = argc <= 3 ? 40: std::atoi(argv[3]);
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
    auto start = t();
    blaze::DynamicMatrix<FLOAT_TYPE> mat(n, nd);
    OMP_PRAGMA("omp parallel for")
    for(auto i = 0u; i < n; ++i) {
        auto r = row(mat, i);
        std::memcpy(&r[0], &ptr[i][0], sizeof(FLOAT_TYPE) * nd);
    }
    auto centers = clustering::kmeanspp(ptr, ptr + n, gen, npoints);
    auto kc = clustering::kcenter_greedy_2approx(ptr, ptr + n, gen, npoints);
    auto centers2 = clustering::kmeanspp(mat, gen, npoints, blz::L1Norm());
    auto matrowit = blz::rowiterator(mat);
    if(0) {
        auto cs = clustering::outliers::kcenter_coreset(matrowit.begin(), matrowit.end(), gen, 3, 0.5);
        auto csmat = index2matrix(cs, mat);
    }
    auto stop = t();
    std::fprintf(stderr, "Time: %gs\n", double((stop - start).count()) / 1e9);
    //for(const auto v: centers) std::fprintf(stderr, "Woo: %u\n", v);
    std::destroy_n(ptr, n);
    std::free(ptr);
}
