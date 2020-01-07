#include "fgc/jv.h"

int main() {

    fgc::Graph<boost::undirectedS, float> g;
    std::vector<typename fgc::Graph<boost::undirectedS, float>::vertex_descriptor> vxs(g.vertices().begin(), g.vertices().end());
    if(0) {
        fgc::jain_vazirani_kmedian(g, vxs, 15);
    }
    std::fprintf(stderr, "Getting here only checks compilation, not correctness, of JV draft.\n");
    size_t dim = 20;
    size_t np = 200;
    blaze::DynamicMatrix<float> points(np, dim);
#if 0
    std::mt19937_64 mt;
    std::uniform_real_distribution<float> urd;
    for(auto r: blz::rowiterator(points))
        for(auto &v: r)
            v = urd(mt);
#endif
    randomize(points);
    points = 1. / points;
    std::set<int> indices;
    while(indices.size() < 20)
        indices.insert(std::rand() % points.rows());
    std::vector<int> iv(indices.begin(), indices.end());
    blaze::DynamicMatrix<float> facilities = rows(points, iv);
    assert(facilities.rows() == 20);
    blaze::DynamicMatrix<float> dists(facilities.rows(), points.rows());
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0; i < facilities.rows(); ++i)
        for(size_t j = 0; j < points.rows(); ++j)
            dists(i, j) = blz::l2Dist(row(facilities, i), row(points, j));
    fgc::NaiveJVSolver<float> njv(dists.rows(), dists.columns());
    njv.setup(dists);
    auto res = njv.ufl(dists, 1.5);
    std::fprintf(stderr, "res size: %zu\n", res.size());
    for(int i = 100000; i--; std::fputc('-', stderr));
    auto kmedsol = njv.kmedian(dists, 5);
    assert(kmedsol.size() == 5);
}
