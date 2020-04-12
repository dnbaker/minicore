#include "fgc/jv.h"
#include "fgc/distance.h"

void disptime(std::chrono::time_point<std::chrono::high_resolution_clock> s, std::chrono::time_point<std::chrono::high_resolution_clock> e,
              std::string label="") {
    std::fprintf(stderr, "%s: %g ms\n", label.data(), (e - s).count() * 1.e-6);
}
#define float double

int main(int argc, char *argv[]) {
    if(std::find_if(argv, argv + argc, [](auto x){return std::strcmp(x, "-h") == 0 || std::strcmp(x, "--help") == 0;}) != argv + argc) {
        std::fprintf(stderr, "Usage: %s [npoints=3000] [nfac=400] [k=50]\nAll arguments optional\n", argv[0]);
        std::exit(1);
    }
    size_t dim = 50;
    size_t np =  argc == 1 ? 3000: std::atoi(argv[1]);
    size_t nf =  argc <= 2 ? 400: std::atoi(argv[2]);
    unsigned k = argc <= 3 ? 50: std::atoi(argv[3]);
    wy::WyRand<uint32_t, 2> rng(13);
    std::uniform_real_distribution<float> gen(0);
    blaze::DynamicMatrix<float> points = blaze::generate(np, dim, [&](auto,auto){return 1. / gen(rng);});
    //std::cout << "Points: " << points << '\n';
    std::set<int> indices;
    while(indices.size() < nf)
        indices.insert(std::rand() % points.rows());
    std::vector<int> iv(indices.begin(), indices.end());
    blaze::DynamicMatrix<float> facilities = rows(points, iv);
    assert(facilities.rows() == nf);
    blaze::DynamicMatrix<float> dists(facilities.rows(), points.rows());
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0; i < facilities.rows(); ++i)
        for(size_t j = 0; j < points.rows(); ++j)
            dists(i, j) = blz::l2Dist(row(facilities, i), row(points, j));
    //std::cout << "dists: " << dists << '\n';
    //disptime(to, eo, "JV old ufl");
    //std::fprintf(stderr, "res size: %zu\n", res.size());
    auto t = std::chrono::high_resolution_clock::now();
    fgc::jv::JVSolver<blaze::DynamicMatrix<float>, float, uint32_t> jvs(dists, 5903.483329773);
    auto t2 = std::chrono::high_resolution_clock::now();
    jvs.run();
    disptime(t, t2, "JV new");
    jvs.make_verbose();
    std::fprintf(stderr, "jvs solution cost: %g\n", jvs.calculate_cost(false));
    auto [kmedcenters, kmedasns] = jvs.kmedian(k, 1000);
    assert(kmedcenters.size() == k);
    fgc::jv::JVSolver<blaze::DynamicMatrix<float>, float, uint32_t> jvs2(jvs, 14000);
    auto jvs2sol = jvs2.run();
}
