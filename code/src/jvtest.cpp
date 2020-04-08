#include "fgc/jv.h"
#include "fgc/distance.h"

void disptime(std::chrono::time_point<std::chrono::high_resolution_clock> s, std::chrono::time_point<std::chrono::high_resolution_clock> e,
              std::string label="") {
    std::fprintf(stderr, "%s: %g ms\n", label.data(), (e - s).count() * 1.e-6);
}

int main() {
    if(0) {
        fgc::Graph<boost::undirectedS, float> g;
        std::vector<typename fgc::Graph<boost::undirectedS, float>::vertex_descriptor> vxs(g.vertices().begin(), g.vertices().end());
        fgc::jain_vazirani_kmedian(g, vxs, 15);
    }
    std::fprintf(stderr, "Getting here only checks compilation, not correctness, of JV draft.\n");
    size_t dim = 50;
    size_t np = 3000;
    unsigned k = 50, nf = 400;
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
    std::fprintf(stderr, "jvs solution cost: %g\n", jvs.calculate_cost(false));
    auto [kmedcenters, kmedasns] = jvs.kmedian(k, 1000);
    assert(kmedcenters.size() == k);
}
