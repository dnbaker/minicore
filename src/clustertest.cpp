#include "blaze/math/DynamicMatrix.h"
#include "aesctr/wy.h"
#include "minocore/wip/clustering.h"

using namespace minocore;

#if 0
int matrix_main() {
    using MatrixMeta = clustering::Meta<
        float, uint32_t,
        clustering::HARD, clustering::INTRINSIC,
        clustering::CONSTANT_FACTOR, clustering::THORUP_SAMPLING,
        clustering::METRIC_KMEDIAN>;
    blz::DM<float> dm = blaze::generate(1000, 1000, [](auto, auto) {return double(std::rand()) / RAND_MAX;});
    dm *= 10;
    blz::DV<uint32_t> selection{1,2,3,4,5,6,7,8,9,900};
    auto oracle = clustering::make_lookup_data_oracle(dm);
    clustering::ClusteringSolverBase<decltype(oracle), MatrixMeta> solver(oracle, 1000, 10);
    solver.set_centers(std::move(selection));
    solver.set_assignments_and_costs();
    const auto &asn = solver.get_assignments(false);
    //std::cerr << asn << '\n';
    return 0;
}

int kmeans_main() {
    unsigned k = 10;
    using MatrixMeta = clustering::Meta<
        float, uint32_t,
        clustering::HARD, clustering::EXTRINSIC,
        clustering::CONSTANT_FACTOR, clustering::D2_SAMPLING,
        clustering::EXPECTATION_MAXIMIZATION>;
    blz::DM<float> dm = blaze::generate(1000, 1000, [](auto, auto) {return double(std::rand()) / RAND_MAX;});
    blz::DV<uint32_t> selection{1,2,3,4,5,6,7,8,9,900};
    auto oracle = clustering::make_exfunc_oracle(dm, blz::sqrL2Norm());
    clustering::ClusteringSolverBase<decltype(oracle), MatrixMeta> solver(oracle, 1000, k);
    solver.approx_sol();
#if 0
    wy::WyRand<uint32_t, 4> rng(10);
    auto [centerids, ogasn, dists] = coresets::kmeanspp(dm, rng, k, blz::sqrL2Norm());
    std::vector<blz::DV<float, blz::rowVector>> centers;
    centers.reserve(k);
    for(const auto cid: centerids) {
        centers.emplace_back(row(dm, cid));
    }
    solver.set_centers(std::move(centers));
#endif
    solver.set_assignments_and_costs();
    const auto &asn = solver.get_assignments(false);
    std::cerr << asn << '\n';
    return 0;
}
#endif

int main() {
    int ret = 0;
    if(0) {
        blz::DM<float> dm= blaze::generate(100, 100, [](auto,auto){return 4;});
        auto jsdapp = make_probdiv_applicator(dm, blz::SQRL2);
        clustering::perform_clustering<clustering::HARD, clustering::EXTRINSIC>(jsdapp, 10);
    }
#if 0
    if((ret = matrix_main())) return ret;
    if((ret = kmeans_main())) return ret;
#endif
    return 0;
}
