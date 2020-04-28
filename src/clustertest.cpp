#include "blaze/math/DynamicMatrix.h"
#include "aesctr/wy.h"
#include "minocore/wip/clustering.h"

using namespace minocore;

int main() {
    using MatrixMeta = clustering::Meta<
        float, uint32_t,
        clustering::HARD, clustering::INTRINSIC,
        clustering::CONSTANT_FACTOR, clustering::THORUP_SAMPLING,
        clustering::METRIC_KMEDIAN>;
    blz::DM<float> dm = blaze::generate(1000, 1000, [](auto x, auto y) {return double(std::rand()) / RAND_MAX;});
    dm *= 10;
    blz::DV<uint32_t> selection{1,2,3,4,5,6,7,8,9,900};
    auto oracle = clustering::make_lookup_data_oracle(dm);
    clustering::ClusteringSolverBase<decltype(oracle), MatrixMeta> solver(oracle, 1000, 10);
    solver.set_centers(std::move(selection));
    solver.set_assignments_and_costs();
    const auto &asn = solver.get_assignments(false);
    std::cerr << asn << '\n';
}
