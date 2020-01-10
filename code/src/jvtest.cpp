#include "fgc/jv.h"
//#include "fgc/blaze_adaptor.h"

int main() {

    fgc::Graph<boost::undirectedS, float> g;
    std::vector<typename fgc::Graph<boost::undirectedS, float>::vertex_descriptor> vxs(g.vertices().begin(), g.vertices().end());
    if(0) {
        fgc::jain_vazirani_kmedian(g, vxs, 15);
    }
    std::fprintf(stderr, "Getting here only checks compilation, not correctness, of JV draft.\n");
    size_t dim = 50;
    size_t np = 800;
    unsigned k = 15, nf = dim;
    wy::WyRand<uint32_t, 2> rng(13);
    std::uniform_real_distribution<float> gen(dim * np);
    blaze::DynamicMatrix<float> points(np, dim);
#if 0
    std::mt19937_64 mt;
    std::uniform_real_distribution<float> urd;
    for(auto r: blz::rowiterator(points))
        for(auto &v: r)
            v = urd(mt);
#endif
    for(auto r: blz::rowiterator(points))
        for(auto &v: r)
            v = gen(rng);
    points = 1. / points;
    std::set<int> indices;
    while(indices.size() < nf)
        indices.insert(std::rand() % points.rows());
    std::vector<int> iv(indices.begin(), indices.end());
    blaze::DynamicMatrix<float> facilities = rows(points, iv);
    assert(facilities.rows() == dim);
    blaze::DynamicMatrix<float> dists(facilities.rows(), points.rows());
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0; i < facilities.rows(); ++i)
        for(size_t j = 0; j < points.rows(); ++j)
            dists(i, j) = blz::l2Dist(row(facilities, i), row(points, j));
    fgc::NaiveJVSolver<float> njv(dists.rows(), dists.columns());
    njv.setup(dists);
    auto res = njv.ufl(dists, 1.5);
    std::fprintf(stderr, "res size: %zu\n", res.size());
    auto kmedsol = njv.kmedian(dists, k);
    assert(kmedsol.size() == k);
}
