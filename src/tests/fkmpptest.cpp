#include "minicore/optim/kmeans.h"
#include "minicore/optim/kcenter.h"
#include "minicore/coreset/kcenter.h"
#include "minicore/dist/applicator.h"
#include <new>
#include <chrono>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

auto t() {return std::chrono::high_resolution_clock::now();}

#ifdef FLOAT_TYPE
#error("DO NOT DEFINE FLOAT_TYPE")
#endif
#define FLOAT_TYPE float


using namespace minicore;
using namespace minicore::coresets;

template<typename Mat, typename RNG>
void test_kccs(Mat &mat, RNG &rng, size_t npoints, double eps) {
    auto matrowit = blz::rowiterator(mat);
    auto start = t();
    double gamma = 500. / mat.rows();
    if(gamma >= 0.5)
        gamma = 0.05;
    auto cs = kcenter_coreset_outliers<decltype(matrowit.begin()), FLOAT_TYPE>(matrowit.begin(), matrowit.end(), rng, npoints, eps,
                /*mu=*/0.5, 1.5, gamma);
    auto maxv = *std::max_element(cs.indices_.begin(), cs.indices_.end());
    std::fprintf(stderr, "max index: %u\n", unsigned(maxv));
    auto stop = t();
    std::fprintf(stderr, "kcenter coreset took %0.12gms\n", util::timediff2ms(stop, start));
    start = t();
    auto csmat = index2matrix(cs, mat);
    stop = t();
    std::fprintf(stderr, "kcenter compacting to coreset took %0.12gs\n", util::timediff2ms(stop, start));
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
#endif
    std::srand(0);
    size_t n = argc == 1 ? 25000: std::atoi(argv[1]);
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
    std::fprintf(stderr, "Time for kmeans++: %0.12gs\n", double((stop - start).count()) / 1e9);
    std::fprintf(stderr, "cost for kmeans++: %0.12g\n", std::accumulate(std::get<2>(centers).begin(), std::get<2>(centers).end(), 0.));
    start = t();
    auto centers3 = reservoir_kmeanspp(ptr, ptr + n, gen, npoints, blz::sqrL2Norm(), (double *)nullptr, 0, 1);

    // WFT *weights=static_cast<WFT *>(nullptr), bool multithread=true, int lspprounds=0, int ntimes=1
    stop = t();
    std::fprintf(stderr, "Time for reservoir_kmeans++: %0.12gs\n", double((stop - start).count()) / 1e9);
    std::fprintf(stderr, "cost for reservoir_kmeans++: %0.12g\n", std::accumulate(std::get<2>(centers3).begin(), std::get<2>(centers3).end(), 0.));

    // centers contains [centers, assignments, distances]
    start = t();
    auto kmc2_centers = kmc2(ptr, ptr + n, gen, npoints, 200);
    stop = t();
    std::fprintf(stderr, "Time for kmc^2: %0.12gs\n", double((stop - start).count()) / 1e9);
    auto [kmcasn, kmccosts] = get_oracle_costs([&](size_t i, size_t j) {
        return blz::sqrL2Dist(ptr[i], ptr[j]);
    }, n, kmc2_centers);
    std::fprintf(stderr, "cost for kmc2: %0.12g\n", blz::sum(kmccosts));
    // then, a coreset can be constructed
    start = t();
    auto kc = kcenter_greedy_2approx(ptr, ptr + n, gen, npoints);
#ifndef NDEBUG
    assert(std::set<uint32_t>(kc.begin(), kc.end()).size() == npoints);
#else
    auto v = std::set<uint32_t>(kc.begin(), kc.end()).size();
    if(v != npoints) {
        std::cerr << v << "instead of " << npoints << '\n';
        throw std::runtime_error("Wrong number of points in k-center algorithm");
    }
#endif
    stop = t();
    std::fprintf(stderr, "Time for kcenter_greedy_2approx: %0.12gs\n", double((stop - start).count()) / 1e9);
    start = t();
    auto centers2 = kmeanspp(mat, gen, npoints, blz::L1Norm(), true, (FLOAT_TYPE *)nullptr, std::ceil(std::log(npoints)));
    stop = t();
    std::fprintf(stderr, "Time for kmeans++ on L1 norm on matrix: %0.12gs\n", double((stop - start).count()) / 1e9);
    test_kccs(mat, gen, npoints, eps);
    //for(const auto v: centers) std::fprintf(stderr, "Woo: %u\n", v);
    start = t();
    auto kmppmcs = kmeans_matrix_coreset(mat, npoints, gen, std::min(size_t(npoints * 10), mat.rows() / 2));
    stop = t();
    std::fprintf(stderr, "Time for kmeans++ matrix coreset: %0.12gs\n", double((stop - start).count()) / 1e9);
    OMP_ONLY(omp_set_num_threads(1);)
    blaze::DynamicMatrix<FLOAT_TYPE> sqmat(20, 20);
    randomize(sqmat);
    sqmat = map(sqmat, [](auto x) {return x * x + 1e-15;});
    assert(min(sqmat) > 0.);
    {
        auto greedy_metric = kcenter_greedy_2approx(blz::rowiterator(sqmat).begin(), blz::rowiterator(sqmat).end(),
                                                    gen, /*k=*/npoints, MatrixLookup{});
        kcenter_greedy_2approx_outliers(blz::rowiterator(sqmat).begin(), blz::rowiterator(sqmat).end(), gen, /*k=*/npoints, eps, .001, MatrixLookup{});
    }
    auto kmpp_asn = std::move(std::get<1>(centers));
    std::vector<FLOAT_TYPE> counts(npoints);
    blz::DynamicMatrix<FLOAT_TYPE> centermatrix(std::get<0>(centers).size(), nd);
    for(unsigned i = 0; i < std::get<0>(centers).size(); ++i) {
        row(centermatrix, i) = row(mat, std::get<0>(centers)[i]);
    }
    double tolerance = 0;
    decltype(centermatrix) copy_mat(centermatrix);
    const unsigned maxrounds = 10000;
    double fulldata_cost = lloyd_loop(kmpp_asn, counts, centermatrix, mat, tolerance, maxrounds);
    double fulldata_cost_ma = lloyd_loop(kmpp_asn, counts, centermatrix, mat, tolerance, maxrounds, sqrL2Norm(), (FLOAT_TYPE *)nullptr, true);
    double fulldata_cost_vanilla = lloyd_loop(kmpp_asn, counts, centermatrix, mat, tolerance, maxrounds, sqrL2Norm(), (FLOAT_TYPE *)nullptr, false);
    std::fprintf(stderr, "Cost for fulldata (normal lloyd) %0.12g vs moving average %0.12g for a difference of %0.12g (and with vanilla on top of ma %0.12g/%0.12g less than the minima of the others)\n",
                 fulldata_cost, fulldata_cost_ma, fulldata_cost - fulldata_cost_ma, fulldata_cost_vanilla, std::min(fulldata_cost_ma, fulldata_cost) - fulldata_cost_vanilla);
    if(npoints > kmppmcs.mat_.rows()) npoints = kmppmcs.mat_.rows();
    auto [wcenteridx, wasn, wcosts] = kmeanspp(kmppmcs.mat_, gen, npoints, blz::sqrL2Norm(), kmppmcs.weights_.data());
    blaze::DynamicMatrix<FLOAT_TYPE> weight_kmppcenters = blz::rows(kmppmcs.mat_, wcenteridx.data(), wcenteridx.size());
    std::fprintf(stderr, "About to perform weighted kmeans\n");
    lloyd_loop(wasn, counts, weight_kmppcenters, kmppmcs.mat_, 0., 1000, blz::sqrL2Norm(), kmppmcs.weights_.data());
    double cost = 0.;
    for(size_t i = 0; i < mat.rows(); ++i) {
        auto mr = row(mat, i);
        FLOAT_TYPE rc = blz::sqrL2Dist(mr, row(weight_kmppcenters, 0));
        for(unsigned j = 1; j < weight_kmppcenters.rows(); ++j)
            rc = std::min(rc, blz::sqrL2Dist(mr, row(weight_kmppcenters, j)));
        cost += rc;
    }
    std::fprintf(stderr, "Cost of coreset solution: %0.12g. Cost of solution on full dataset: %0.12g\n", cost, fulldata_cost);
}
