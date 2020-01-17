#include "fgc/graph.h"
#include "fgc/parse.h"
#include "fgc/bicriteria.h"
#include "fgc/coreset.h"
#include "fgc/lsearch.h"
#include "fgc/jv.h"
#include <ctime>

template<typename T> class TD;


using namespace fgc;
using namespace boost;



#if 0
fgc::Graph<undirectedS> &
max_component(fgc::Graph<undirectedS> &g) {
#endif
template<typename GraphT>
GraphT &
max_component(GraphT &g) {
    auto ccomp = std::make_unique<uint32_t []>(boost::num_vertices(g));
    assert(&ccomp[0] == ccomp.get());
    unsigned ncomp = boost::connected_components(g, ccomp.get());
    if(ncomp != 1) {
        std::fprintf(stderr, "not connected. ncomp: %u\n", ncomp);
        std::vector<unsigned> counts(ncomp);
        for(size_t i = 0, e = boost::num_vertices(g); i < e; ++counts[ccomp[i++]]);
        auto maxcomp = std::max_element(counts.begin(), counts.end()) - counts.begin();
        std::fprintf(stderr, "maxcmp %zu out of total %u\n", maxcomp, ncomp);
        flat_hash_map<uint64_t, uint64_t> remapper;
        size_t id = 0;
        for(size_t i = 0; i < boost::num_vertices(g); ++i) {
            if(ccomp[i] == maxcomp) {
                remapper[i] = id++;
            }
        }
        GraphT newg(counts[maxcomp]);
        typename boost::property_map <fgc::Graph<undirectedS>,
                             boost::edge_weight_t >::type EdgeWeightMap = get(boost::edge_weight, g);
        for(auto edge: g.edges()) {
            auto lhs = source(edge, g);
            auto rhs = target(edge, g);
            if(auto lit = remapper.find(lhs), rit = remapper.find(rhs);
               lit != remapper.end() && rit != remapper.end()) {
                boost::add_edge(lit->second, rit->second, EdgeWeightMap[edge], newg);
            }
        }
        ncomp = boost::connected_components(newg, ccomp.get());
        std::fprintf(stderr, "num components: %u. num edges: %zu. num nodes: %zu\n", ncomp, newg.num_edges(), newg.num_vertices());
        assert(ncomp == 1);
        std::swap(newg, g);
    }
    return g;
}

int main(int argc, char **argv) {
    std::string input = argc == 1 ? "../data/dolphins.graph": const_cast<const char *>(argv[1]);
    const unsigned k = argc > 2 ? std::atoi(argv[2]): 12;
    std::srand(std::hash<std::string>{}(input));
    std::string fn = std::string("default_scratch.") + std::to_string(std::rand()) + ".tmp";
    const double z = 1.; // z = power of the distance norm
    if(argc > 3) fn = argv[3];

    fgc::Graph<undirectedS, float> g = parse_by_fn(input);
    max_component(g);
    // Assert that it's connected, or else the problem has infinite cost.
    uint64_t seed = 1337;

    size_t nsampled_max = std::min(std::ceil(std::pow(std::log2(boost::num_vertices(g)), 2.5)), 3000.);
    if(nsampled_max > boost::num_vertices(g))
        nsampled_max = boost::num_vertices(g) / 2;
    auto dm = graph2diskmat(g, fn);
    if(z != 1.) {
        assert(z > 1.);
        ~dm = pow(abs(~dm), z);
    }

    // Perform Thorup sample before JV method.
    auto lsearcher = make_kmed_lsearcher(~dm, k, 1e-5, seed);
    lsearcher.run();
    auto med_solution = lsearcher.sol_;
    auto ccost = lsearcher.current_cost_;
    std::fprintf(stderr, "cost: %f\n", ccost);
    // Calculate the costs of this solution
    auto [costs, assignments] = get_costs(g, med_solution);
    if(z != 1.)
        costs = blaze::pow(blaze::abs(costs), z);
    // Build a coreset importance sampler based on it.
    coresets::CoresetSampler<float, uint32_t> sampler;
    sampler.make_sampler(costs.size(), med_solution.size(), costs.data(), assignments.data());
    auto sampled_cs = sampler.sample(50);
    std::FILE *ofp = std::fopen("sampler.out", "wb");
    sampler.write(ofp);
    std::fclose(ofp);
}
