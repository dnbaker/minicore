#include "graph.h"
#include "parse.h"
#include "bicriteria.h"
#include "coreset.h"
#include "lsearch.h"
#include "jv.h"

template<typename T> class TD;


using namespace fgc;
using namespace boost;

#define undirectedS bidirectionalS

auto dimacs_official_parse(std::string input) {
    fgc::Graph<undirectedS> g;
    std::ifstream ifs(input);
    std::string graphtype;
    size_t nnodes = 0, nedges = 0;
    for(std::string line; std::getline(ifs, line);) {
        if(line.empty()) continue;
        switch(line.front()) {
            case 'c': break; // nothing
            case 'p': {
                const char *p = line.data() + 2, *p2 = ++p;
                while(!std::isspace(*p2)) ++p2;
                graphtype = std::string(p, p2 - p);
                std::fprintf(stderr, "graphtype: %s\n", graphtype.data());
                p = p2 + 1;
                nnodes = std::strtoull(p, nullptr, 10);
                for(size_t i = 0; i < nnodes; ++i)
                    boost::add_vertex(g); // Add all the vertices
                if((p2 = std::strchr(p, ' ')) == nullptr) throw std::runtime_error(std::string("Failed to parse file at ") + input);
                p = p2 + 1;
                nedges = std::strtoull(p, nullptr, 10);
                std::fprintf(stderr, "n: %zu. m: %zu\n", nnodes, nedges);
                break;
            }
            case 'a': {
                assert(nnodes);
                char *strend;
                const char *p = line.data() + 2;
                size_t lhs = std::strtoull(p, &strend, 10);
                p = strend + 1;
                size_t rhs = std::strtoull(p, &strend, 10);
                p = strend + 1;
                double dist = std::atof(p);
                assert(lhs >= 1);
                assert(rhs >= 1);
                boost::add_edge(lhs - 1, rhs - 1, dist, g);
                break;
            }
            default: std::fprintf(stderr, "Unexpected: this line! (%s)\n", line.data()); throw std::runtime_error("");
        }
    }
    return g;
}

auto dimacs_parse(const char *fn) {
    auto g = parse_dimacs_unweighted<boost::undirectedS>(fn);
    using Graph = decltype(g);
    boost::graph_traits<decltype(g)>::edge_iterator ei, ei_end;
    wy::WyRand<uint64_t, 2> gen(boost::num_vertices(g));
    for(std::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
        boost::put(boost::edge_weight_t(), g, *ei, 1. / (double(gen()) / gen.max()));
    }
    for(auto [vs, ve] = boost::vertices(g); vs != ve; ++vs) {
        boost::graph_traits<Graph>::vertex_descriptor vind = *vs;
        ++vind;
    }
    return g;
}
auto csv_parse(const char *fn) {
    return parse_nber<boost::undirectedS>(fn);
}

int main(int argc, char **argv) {
#ifdef _OPENMP
    omp_set_num_threads(std::thread::hardware_concurrency());
#endif
    std::string input = argc == 1 ? "../data/dolphins.graph": const_cast<const char *>(argv[1]);
    const unsigned k = argc > 2 ? std::atoi(argv[2]): 12;
    std::string fn = "default_scratch.fn";
    if(argc > 3) fn = argv[3];
    
    fgc::Graph<undirectedS> g;
    // 1. Parse
    if(input.find(".csv") != std::string::npos) {
        g = csv_parse(input.data());
    } else if(input.find(".gr") != std::string::npos && input.find(".graph") == std::string::npos) {
        g = dimacs_official_parse(input);
    } else {
        g = dimacs_parse(input.data());
    }
    // Assert that it's connected, or else the problem has infinite cost.
    auto ccomp = std::make_unique<uint32_t []>(boost::num_vertices(g));
    assert(&ccomp[0] == ccomp.get());
    unsigned ncomp = boost::connected_components(g, ccomp.get());
    if(ncomp != 1) {
        std::fprintf(stderr, "not connected. ncomp: %u\n", ncomp);
        return 1;
    }
    ccomp.reset();
    assert(boost::num_vertices(g) > k);
    uint64_t seed = 1337;

    size_t nsampled_max = std::min(std::ceil(std::pow(std::log2(boost::num_vertices(g)), 2.5)), 3000.);
    if(nsampled_max > boost::num_vertices(g))
        nsampled_max = boost::num_vertices(g) / 2;
    auto dm = graph2diskmat(g, fn);

    // Perform Thorup sample before JV method.
#if 0
    double frac = nsampled_max / double(boost::num_vertices(g));
    auto sampled = thorup_sample(g, k, seed, frac); // 0 is the seed, 500 is the maximum sampled size
    std::fprintf(stderr, "sampled size: %zu\n", sampled.size());
    std::fprintf(stderr, "ncomp: %u\n", ncomp);

    // Use this sample to generate an approximate k-median solution
    auto med_solution = fgc::jain_vazirani_kmedian(g, sampled, k);

    // Sanity check: make sure that our sampled points are in the Thorup sample.
#ifndef NDEBUG
    for(const auto v: med_solution) assert(std::find(sampled.begin(), sampled.end(), v) != sampled.end());
#endif

    std::fprintf(stderr, "med solution size: %zu\n", med_solution.size());
#else
    int nr = 100;
    unsigned nseedings = 10;
    auto lsearcher = make_kmed_lsearcher(~dm, k, 1e-5, seed);
    lsearcher.run(nr);
    size_t nreplaced = 0;
    auto med_solution = lsearcher.sol_;
    auto ccost = lsearcher.current_cost_;
    for(size_t i = 0; i < nseedings; ++i) {
        lsearcher.reseed(seed + i, i % 2);
        lsearcher.run(nr);
        std::fprintf(stderr, "old cost: %g. new cost: %g\n", ccost, lsearcher.current_cost_);
        std::fprintf(stderr, "cost for sol: %g\n", lsearcher.cost_for_sol(lsearcher.sol_));
        if(lsearcher.current_cost_ < ccost) {
            std::fprintf(stderr, "replacing with seeding number %u!\n", nseedings);
            med_solution = lsearcher.sol_;
            ccost = lsearcher.current_cost_;
            ++nreplaced;
        }
    }
    std::fprintf(stderr, "nreplaced: %zu/%u\n", nreplaced, nseedings);
#endif
    // Calculate the costs of this solution
    auto [costs, assignments] = get_costs(g, med_solution);
    // Build a coreset importance sampler based on it.
    coresets::CoresetSampler<float, uint32_t> sampler;
    sampler.make_sampler(costs.size(), med_solution.size(), costs.data(), assignments.data());
    auto sampled_cs = sampler.sample(50);
    std::FILE *ofp = std::fopen("sampler.out", "wb");
    sampler.write(ofp);
    std::fclose(ofp);
}
