#include "fgc/graph.h"
#include "fgc/geo.h"
#include "fgc/parse.h"
#include "fgc/bicriteria.h"
#include "fgc/coreset.h"
#include "fgc/lsearch.h"
#include "fgc/jv.h"
#include "fgc/timer.h"
#include "blaze/util/Serialization.h"

void usage(const char *x) {
    std::fprintf(stderr, "Usage: %s <input.blaze> <input.coreset_sampler> k <optional: subset_indices>\n",
                 x);
    std::exit(1);
}
using namespace fgc;
template<typename Mat, typename Con>
void emit_sol_cost(const Mat &dm, const Con &sol, std::string label) {
    std::fprintf(stderr, "cost for %s: %0.12g\n", label.data(),
                 blz::sum(blz::min<blz::columnwise>(rows(dm, sol))));
}

template<typename Mat, typename RNG>
auto
thorup_d(const Mat &mat, RNG &rng, unsigned k)
{
    static constexpr double EPS = 0.5;
    size_t nr = mat.rows();
    assert(nr = mat.columns());
    double logn = std::log(nr);
    const size_t nperround = std::ceil(21. * k * logn / EPS);
    const size_t maxnumrounds = std::ceil(3. * logn);
    std::vector<size_t> R(mat.rows());
    std::iota(R.begin(), R.end(), size_t(0));
    std::vector<size_t> F;
    F.reserve(std::min(nperround * 5, R.size()));
    blaze::DynamicVector<float, blaze::rowVector> distances(mat.rows(),  std::numeric_limits<float>::max());
    shared::flat_hash_set<size_t> vertices;
    size_t i;
    for(i = 0; R.size() && i < maxnumrounds; ++i) {
        if(R.size() > nperround) {
            do vertices.insert(R[rng() % R.size()]); while(vertices.size() < nperround);
            F.insert(F.end(), vertices.begin(), vertices.end());
            for(const auto v: vertices)
                distances = blaze::min(distances, row(mat, v BLAZE_CHECK_DEBUG));
            vertices.clear();
        } else {
            for(const auto r: R) {
                F.push_back(r);
                distances = blaze::min(distances, row(mat, r BLAZE_CHECK_DEBUG));
            }
            R.clear();
        }
        if(R.empty()) break;
        auto randel = R[rng() % R.size()];
        auto minv = distances[randel];
        R.erase(std::remove_if(R.begin(), R.end(), [d=distances.data(),minv](auto x) {return d[x] <= minv;}), R.end());
    }
    if(i >= maxnumrounds && R.size()) {
        // This failed. Do not use this round.
        return std::make_pair(std::move(F), std::numeric_limits<double>::max());
    }
    return std::make_pair(std::move(F), double(blaze::sum(distances)));
}

template<typename Mat, typename RNG>
auto
thorup_mincost(const Mat &mat, RNG &rng, unsigned k, unsigned ntries) {
    auto ret = thorup_d(mat, rng, k);
    unsigned trynum = 0;
    while(++trynum < ntries) {
        auto nextsamp = thorup_d(mat, rng, k);
        if(nextsamp.second < ret.second)
            std::swap(ret, nextsamp);
        else if(nextsamp.second == std::numeric_limits<float>::max()) --trynum;
    }
    return ret;
}

int main(int argc, char **argv) {
    std::vector<unsigned> coreset_sizes{
        5, 10, 15, 20, 25, 50, 75, 100, 125, 250, 375, 500, 625, 1250, 1875, 2500, 3125, 3750
    };
    if(argc < 4 || argc > 5) usage(argv[0]);
    std::string msg = "'";
    for(auto av = argv; *av; ++av) {
        msg = msg + *av + (av[1] ? ' ': '\'');
    }
    std::fprintf(stderr, "command-line: %s\n", msg.data());
    blaze::DynamicMatrix<float> dm;
    {
        blaze::Archive<std::ifstream> ifs(argv[1]);
        ifs >> dm;
    }
    unsigned k = std::atoi(argv[3]);
    if(k <= 0) throw std::runtime_error("k must be > 0");

    std::srand(std::time(nullptr));
    wy::WyRand<uint32_t, 2> rng(std::rand());
    auto thorup_sampled = thorup_mincost(dm, rng, k, 5);
    blaze::DynamicMatrix<float> thorup_dm(columns(rows(dm, thorup_sampled.first), thorup_sampled.first));
    std::fprintf(stderr, "Thorup sampled down to %zu from %zu\n", thorup_sampled.first.size(), dm.rows());

    blaze::DynamicVector<uint32_t> subset_indices;
    std::fprintf(stderr, "size of dm: %zu/%zu\n", dm.rows(), dm.columns());
    fgc::coresets::CoresetSampler<float, uint32_t> cs;
    cs.read(argv[2]);
    std::fprintf(stderr, "size of coreset sampler: %zu\n", cs.size());
#if 0
    if(std::ifstream isfile(argv[4]); isfile) {
        blaze::Archive<std::ifstream> ifs(argv[4]);
        ifs >> subset_indices;
    }
    if(subset_indices.size())
        std::fprintf(stderr, "size of selected indices: %zu\n", subset_indices.size());
#endif
    auto lsearcher = fgc::make_kmed_lsearcher(dm, k, 1e-3, 13);
    util::Timer timer("full local search");
    std::vector<uint32_t> fullsol, thorup_sol;
    lsearcher.run();
    timer.report();
    fullsol.assign(lsearcher.sol_.begin(), lsearcher.sol_.end());
    emit_sol_cost(dm, fullsol, "Full cost");
    auto thorup_lsearcher = fgc::make_kmed_lsearcher(thorup_dm, k, 1e-3, 13);
    timer.restart("Thorup local search");
    thorup_lsearcher.run();
    thorup_sol.assign(thorup_lsearcher.sol_.begin(), thorup_lsearcher.sol_.end());
    emit_sol_cost(dm, thorup_sol, "Thorup");
    timer.report();
    timer.reset();
    for(const auto coreset_size: coreset_sizes) {
        auto csstr = std::to_string(coreset_size);
        timer.restart(std::string("coreset sample: ") + csstr);
        auto coreset = cs.sample(coreset_size);
        assert(coreset.indices_.data());
        assert(coreset.weights_.data());
        for(const auto idx: coreset.indices_) {
            assert(idx < dm.rows() || !std::fprintf(stderr, "idx: %u\n", unsigned(idx)));
        }
        timer.report();
        blaze::DynamicMatrix<float> coreset_dm(columns(dm, coreset.indices_.data(), coreset.indices_.size()));
        std::fprintf(stderr, "Made coreset dm. cols: %zu. rows: %zu\n", coreset_dm.columns(), coreset_dm.rows());
        std::vector<uint32_t> vxs_sol, sxs_sol;
        for(unsigned i = 0; i < coreset_dm.columns(); ++i)
            column(coreset_dm, i) *= coreset.weights_[i];
        std::fprintf(stderr, "reweighted. Optimizing\n");
        auto lsearcher = fgc::make_kmed_lsearcher(coreset_dm, k, 1e-3, 13);
        timer.restart(std::string("optimize over V x S") + csstr);
        lsearcher.run();
        timer.report();
        vxs_sol.assign(lsearcher.sol_.begin(), lsearcher.sol_.end());
        emit_sol_cost(dm, vxs_sol, std::string("vxs") + csstr);
        coreset_dm = rows(coreset_dm, coreset.indices_.data(), coreset.indices_.size());
        auto &cd = coreset_dm;
        assert(cd.rows() == cd.columns());
        std::fprintf(stderr, "cd rows: %zu\n", cd.rows());
        auto sxs_searcher = fgc::make_kmed_lsearcher(coreset_dm, k, 1e-3, 13);
        timer.restart(std::string("optimize over S x S") + csstr);
        sxs_searcher.run();
        timer.report();
        sxs_sol.assign(sxs_searcher.sol_.begin(), sxs_searcher.sol_.end());
        for(auto &i: sxs_sol) {
            std::fprintf(stderr, "sol is %u of %zu\n", i, coreset_dm.rows());
            i = coreset.indices_.at(i);
        }
        emit_sol_cost(dm, sxs_sol, std::string("sxs") + csstr);
    }
}
