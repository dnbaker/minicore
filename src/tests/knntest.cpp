#include "include/minocore/dist/knngraph.h"

int main() {
    blaze::DynamicMatrix<float> mat = blaze::generate(1000, 50, [](auto x, auto y) {
        return float(std::rand()) / RAND_MAX + (x * y) / 1000. / 50.;
    });
    auto app = minocore::jsd::make_probdiv_applicator(mat, blz::distance::L1);
    auto knns = minocore::make_knns(app, 10);
    auto graph = minocore::knns2graph(knns, app.size(), true);
    auto mst = minocore::knng2mst(graph);
    std::fprintf(stderr, "mst size: %zu edges vs %zu nodes\n", mst.size(), app.size());
}
