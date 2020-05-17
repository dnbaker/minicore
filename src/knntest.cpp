#include "include/minocore/dist/knngraph.h"

int main() {
    blaze::DynamicMatrix<float> mat = blaze::generate(1000, 50, [](auto x, auto y) {
        return float(std::rand()) / RAND_MAX + (x * y) / 1000. / 50.;
    });
    auto app = minocore::jsd::make_probdiv_applicator(mat);
    auto graph = minocore::make_knn_graph(app, 10, true);
}
