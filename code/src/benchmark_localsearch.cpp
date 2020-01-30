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

    wy::WyRand<uint32_t, 2> rng(std::hash<std::string>()(msg));
    auto thorup_sampled = thorup_mincost(dm, rng, k, 8);
    if(thorup_sampled.first.size() == dm.rows()) throw std::runtime_error("Need to pick a thorup which is smaller\n");
    blaze::DynamicMatrix<float> thorup_dm(columns(rows(dm, thorup_sampled.first), thorup_sampled.first));

    blaze::DynamicVector<size_t> subset_indices;
    std::fprintf(stderr, "size of dm: %zu/%zu\n", dm.rows(), dm.columns());
    fgc::coresets::CoresetSampler<float, uint32_t> cs;
    cs.read(argv[2]);
    std::fprintf(stderr, "size of coreset sampler: %zu\n", cs.size());
    if(std::ifstream isfile(argv[4]); isfile) {
        blaze::Archive<std::ifstream> ifs(argv[4]);
        ifs >> subset_indices;
    }
    if(subset_indices.size())
        std::fprintf(stderr, "size of selected indices: %zu\n", subset_indices.size());
    auto lsearcher = fgc::make_kmed_lsearcher(dm, k, 1e-3, 13);
    util::Timer timer("full local search");
    lsearcher.run();
    timer.report();
    auto thorup_lsearcher = fgc::make_kmed_lsearcher(thorup_dm, k, 1e-3, 13);
    timer.restart("Thorup local search");
    thorup_lsearcher.run();
    timer.report();
    // Next, sample coresets of a set of sizes and measure runtime of Local Search for each of them, along with the cost of their solution
    // Then compare the accuracy to the best found on the full dataset.
}
