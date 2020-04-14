#include "kmeans.h"
#include "kcenter.h"
#include "applicator.h"
#include <new>
#include <chrono>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

#define t std::chrono::high_resolution_clock::now

#ifndef FLOAT_TYPE
#define FLOAT_TYPE double
#endif


using namespace fgc;
using namespace fgc::coresets;

template<typename Mat, typename RNG>
void test_kccs(Mat &mat, RNG &rng, size_t npoints, double eps) {
    auto matrowit = blz::rowiterator(mat);
    auto start = t();
    double gamma = 100. / mat.rows();
    if(gamma >= 0.5)
        gamma = 0.05;
    auto cs = outliers::kcenter_coreset(matrowit.begin(), matrowit.end(), rng, npoints, eps,
                /*mu=*/0.5, 1.5, gamma);
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
    OMP_ONLY(omp_set_num_threads(nt);)
    std::fprintf(stderr, "%d threads used\n", nt);
#endif
    std::srand(0);
    size_t n = argc == 1 ? 250000: std::atoi(argv[1]);
    size_t npoints = argc <= 2 ? 50: std::atoi(argv[2]);
    size_t nd = argc <= 3 ? 40: std::atoi(argv[3]);
    double eps = argc <= 4 ? 0.5: std::atof(argv[3]);
    std::vector<std::vector<FLOAT_TYPE>> tmpvecs(n);
    auto ptr = tmpvecs.data();
    //std::unique_ptr<std::vector<FLOAT_TYPE>[]> stuff(n);
    wy::WyRand<uint32_t, 2> gen;
    OMP_PRAGMA("omp parallel for")
    for(auto i = 0u; i < n; ++i) {
        tmpvecs[i].resize(nd);
        for(auto &e: tmpvecs[i]) e = FLOAT_TYPE(std::rand()) / RAND_MAX;
    }
    std::fprintf(stderr, "generated\n");
    blaze::DynamicMatrix<FLOAT_TYPE> mat(n, nd);
    OMP_PRAGMA("omp parallel for")
    for(auto i = 0u; i < n; ++i) {
        auto r = row(mat, i);
        std::memcpy(&r[0], &(tmpvecs[i][0]), sizeof(FLOAT_TYPE) * nd);
    }
    auto start = t();
    auto centers = kmeanspp(ptr, ptr + n, gen, npoints);
    auto stop = t();
    std::fprintf(stderr, "Time for kmeans++: %gs\n", double((stop - start).count()) / 1e9);
    std::fprintf(stderr, "cost for kmeans++: %g\n", std::accumulate(std::get<2>(centers).begin(), std::get<2>(centers).end(), 0.));

    // centers contains [centers, assignments, distances]
    start = t();
    auto kmc2_centers = kmc2(ptr, ptr + n, gen, npoints, 200);
    stop = t();
    std::fprintf(stderr, "Time for kmc^2: %gs\n", double((stop - start).count()) / 1e9);
    auto kmccosts = get_oracle_costs([&](size_t i, size_t j) {
        return blz::sqrL2Dist(ptr[i], ptr[j]);
    }, n, kmc2_centers);
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
    auto kmppmcs = kmeans_matrix_coreset(mat, npoints, gen, std::min(size_t(npoints * 10), mat.rows() / 2));
    stop = t();
    std::fprintf(stderr, "Time for kmeans++ matrix coreset: %gs\n", double((stop - start).count()) / 1e9);
    OMP_ONLY(omp_set_num_threads(1);)
    blaze::DynamicMatrix<FLOAT_TYPE> sqmat(20, 20);
    randomize(sqmat);
    sqmat = map(sqmat, [](auto x) {return x * x + 1e-15;});
    assert(min(sqmat) > 0.);
    {
        auto greedy_metric = kcenter_greedy_2approx(rowiterator(sqmat).begin(), rowiterator(sqmat).end(),
                                                    gen, /*k=*/3, MatrixLookup{});
    }
    auto kmpp_asn = std::move(std::get<1>(centers));
    std::vector<FLOAT_TYPE> counts(npoints);
    blz::DynamicMatrix<FLOAT_TYPE> centermatrix(std::get<0>(centers).size(), nd);
    for(unsigned i = 0; i < std::get<0>(centers).size(); ++i) {
        row(centermatrix, i) = row(mat, std::get<0>(centers)[i]);
    }
    double tolerance = 1e-4;
    decltype(centermatrix) copy_mat(centermatrix);
    double fulldata_cost = lloyd_loop(kmpp_asn, counts, centermatrix, mat, tolerance, 100);
    if(npoints > kmppmcs.mat_.rows()) npoints = kmppmcs.mat_.rows();
    auto [wcenteridx, wasn, wcosts] = kmeanspp(kmppmcs.mat_, gen, npoints, blz::sqrL2Norm(), true, kmppmcs.weights_.data());
    blaze::DynamicMatrix<FLOAT_TYPE> weight_kmppcenters = blz::rows(kmppmcs.mat_, wcenteridx.data(), wcenteridx.size());
    lloyd_loop(wasn, counts, weight_kmppcenters, kmppmcs.mat_, 0., 1000, blz::sqrL2Norm(), kmppmcs.weights_.data());
    double cost = 0.;
    for(size_t i = 0; i < mat.rows(); ++i) {
        auto mr = row(mat, i);
        double rc = blz::sqrL2Dist(mr, row(weight_kmppcenters, 0));
        for(unsigned j = 1; j < weight_kmppcenters.rows(); ++j)
            rc = std::min(rc, blz::sqrL2Dist(mr, row(weight_kmppcenters, j)));
        cost += rc;
    }
    std::fprintf(stderr, "Cost of coreset solution: %g. Cost of solution on full dataset: %g\n", cost, fulldata_cost);
}
