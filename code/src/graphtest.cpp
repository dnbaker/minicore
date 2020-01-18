#include "fgc/graph.h"
#include "fgc/parse.h"
#include "fgc/bicriteria.h"
#include "fgc/coreset.h"
#include "fgc/lsearch.h"
#include "fgc/jv.h"
#include <ctime>
#include <getopt.h>
#include "blaze/util/Serialization.h"

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

void usage(const char *ex) {
    std::fprintf(stderr, "usage: %s <opts> [input file or ../data/dolphins.graph]\n"
                         "-k\tset k [12]\n"
                         "-c\tAppend coreset size. Default: {100} (if empty)\n"
                         "-s\tPath to write coreset sampler to\n"
                         "-z\tset z [1.]\n",
                 ex);
    std::exit(1);
}

int main(int argc, char **argv) {
    unsigned k = 10;
    double z = 1.; // z = power of the distance norm
    std::string fn = std::string("default_scratch.") + std::to_string(std::rand()) + ".tmp";
    std::vector<unsigned> coreset_sizes;
    for(int c;(c = getopt(argc, argv, "z:s:c:k:h?")) >= 0;) {
        switch(c) {
            case 'k': k = std::atoi(optarg); break;
            case 'z': z = std::atof(optarg); break;
            case 's': fn = optarg; break;
            case 'c': coreset_sizes.push_back(std::atoi(optarg)); break;
            case 'h': default: usage(argv[0]);
        }
    }
    if(coreset_sizes.empty())
        coreset_sizes.push_back(100);
    std::string input = argc == optind ? "../data/dolphins.graph": const_cast<const char *>(argv[optind]);
    std::srand(std::hash<std::string>{}(input));

    fgc::Graph<undirectedS, float> g = parse_by_fn(input);
    max_component(g);
    // Assert that it's connected, or else the problem has infinite cost.
    uint64_t seed = 1337;

    size_t nsampled_max = std::min(std::ceil(std::pow(std::log2(boost::num_vertices(g)), 2.5)), 20000.);
    const double frac = std::max(double(nsampled_max) / boost::num_vertices(g), .5);
    std::vector<typename boost::graph_traits<decltype(g)>::vertex_descriptor> sampled;
    std::vector<typename boost::graph_traits<decltype(g)>::vertex_descriptor> *ptr = nullptr;
    if(boost::num_vertices(g) > 20000) {
        std::fprintf(stderr, "num vtx: %zu. Thorup sampling!\n", boost::num_vertices(g));
        sampled = thorup_sample(g, k, seed, frac);
        ptr = &sampled;
    }
    auto dm = graph2diskmat(g, fn, ptr);
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
    for(const auto ms: med_solution)
        assert(ms < boost::num_vertices(g));
    // Calculate the costs of this solution
    auto [costs, assignments] = get_costs(g, med_solution);
    if(z != 1.)
        costs = blaze::pow(blaze::abs(costs), z);
    // Build a coreset importance sampler based on it.
    coresets::CoresetSampler<float, uint32_t> sampler;
    sampler.make_sampler(costs.size(), med_solution.size(), costs.data(), assignments.data());
    std::FILE *ofp = std::fopen(fn.data(), "wb");
    sampler.write(ofp);
    std::fclose(ofp);
    for(auto coreset_size: coreset_sizes) {
        if(coreset_size > (~dm).rows()) coreset_size = (~dm).rows();
        auto sampled_cs = sampler.sample(coreset_size);
        std::string fn = std::string("sampled.") + std::to_string(coreset_size) + ".matcs";
        //auto subm = submatrix(~dm, 0, 0, coreset_size, (~dm).columns());
        auto subm = blaze::DynamicMatrix<float>(coreset_size, (~dm).columns());
        std::fprintf(stderr, "About to fill distmat with coreset of size %u\n", coreset_size);
        fill_graph_distmat(g, subm, &sampled_cs.indices_);
        // tmpdm has # indices rows, # nodes columns
        auto columnsel = columns(subm, sampled_cs.indices_.data(), sampled_cs.indices_.size());
        assert(columnsel.rows() == columnsel.columns());
        blaze::Archive<std::ofstream> bfp(fn);
        bfp << sampled_cs.indices_ << sampled_cs.weights_ << columnsel;
    }
}
