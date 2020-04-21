#include "minocore/optim/jv.h"
#include "minocore/dist/distance.h"

void disptime(std::chrono::time_point<std::chrono::high_resolution_clock> s, std::chrono::time_point<std::chrono::high_resolution_clock> e,
              std::string label="") {
    std::fprintf(stderr, "%s: %g ms\n", label.data(), (e - s).count() * 1.e-6);
}

int main(int argc, char *argv[]) {
    if(std::find_if(argv, argv + argc, [](auto x){return std::strcmp(x, "-h") == 0 || std::strcmp(x, "--help") == 0;}) != argv + argc) {
        std::fprintf(stderr, "Usage: %s [npoints=3000] [nfac=400] [k=50]\nAll arguments optional\n", argv[0]);
        std::exit(1);
    }
    size_t np =  argc == 1 ? 3000: std::atoi(argv[1]);
    size_t nf =  argc <= 2 ? 400: std::atoi(argv[2]);
    unsigned k = argc <= 3 ? 50: std::atoi(argv[3]);
    size_t dim = argc <= 4 ? 50: std::atoi(argv[4]);
    std::uniform_real_distribution<float> gen(0);
    blaze::DynamicMatrix<float> points = blaze::generate(np, dim, [&](auto x,auto y){
        wy::WyRand<uint32_t, 0> rng((uint64_t(x) << 32) | y);
        return float(1. / gen(rng));}
    );
    //std::cout << "Points: " << points << '\n';
    std::set<int> indices;
    while(indices.size() < nf)
        indices.insert(std::rand() % points.rows());
    std::vector<int> iv(indices.begin(), indices.end());
    blaze::DynamicMatrix<float> facilities = rows(points, iv);
    assert(facilities.rows() == nf);
    blaze::DynamicMatrix<float> dists(facilities.rows(), points.rows());
    auto diststart = std::chrono::high_resolution_clock::now();
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0; i < facilities.rows(); ++i)
        for(size_t j = 0; j < points.rows(); ++j)
            dists(i, j) = blz::l2Dist(row(facilities, i), row(points, j));
    {
        blaze::DynamicMatrix<float> finalpoints(std::move(points));
    }
    auto diststop = std::chrono::high_resolution_clock::now();
    disptime(diststart, diststop, "Distance matrix calculation");
    //std::cout << "dists: " << dists << '\n';
    //disptime(to, eo, "JV old ufl");
    //std::fprintf(stderr, "res size: %zu\n", res.size());
    auto t = std::chrono::high_resolution_clock::now();
    minocore::jv::JVSolver<blaze::DynamicMatrix<float>, float, uint32_t> jvs(dists, 5903.483329773);
    auto t2 = std::chrono::high_resolution_clock::now();
    disptime(t, t2, "JV setup");
    jvs.run();
    t2 = std::chrono::high_resolution_clock::now();
    disptime(t, t2, "JV, setup + computation");
    jvs.make_verbose();
    t = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "jvs solution cost: %g\n", jvs.calculate_cost(false));
    t2 = std::chrono::high_resolution_clock::now();
    disptime(t, t2, "JV, cost calculation");

    t = std::chrono::high_resolution_clock::now();
    auto [kmedcenters, kmedasns] = jvs.kmedian(k, 75);
    t2 = std::chrono::high_resolution_clock::now();
    disptime(t, t2, "JV k-median calculation");
    assert(kmedcenters.size() == k);
    minocore::jv::JVSolver<blaze::DynamicMatrix<float>, float, uint32_t> jvs2(jvs, 14000);
    auto jvs2sol = jvs2.run();
}
