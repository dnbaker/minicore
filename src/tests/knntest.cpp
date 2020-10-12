#include "include/minicore/dist/knngraph.h"
using namespace minicore;

int main() {
    blaze::DynamicMatrix<float> mat = blaze::generate(1000, 50, [](auto x, auto y) {
        return float(std::rand()) / RAND_MAX + (x * y) / 1000. / 50.;
    });
    auto app = minicore::jsd::make_probdiv_applicator(mat, distance::L1);
    auto knns = minicore::make_knns(app, 10);
    auto graph = minicore::knns2graph(knns, app.size(), true);
    auto mst = minicore::knng2mst(graph);
    std::fprintf(stderr, "mst size: %zu edges vs %zu nodes\n", mst.size(), app.size());
}
