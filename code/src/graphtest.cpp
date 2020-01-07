#include "graph.h"
#include "parse.h"
#include "bicriteria.h"
#include "coreset.h"
#include "jv.h"

template<typename T> class TD;


#define undirectedS bidirectionalS
using namespace fgc;
using namespace boost;

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
                boost::add_edge(lhs, rhs, dist, g);
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
    //typedef boost::graph_traits<Graph> GraphTraits;
    //typename boost::property_map<Graph, boost::vertex_index_t>::type index = get(boost::vertex_index, g);
    //typename boost::property_map<Graph, boost::edge_weight_t>::type weight = get(boost::edge_weight, g);
    //typename boost::property_map<Graph, boost::edge_weight_t>::type weightMap = 
    //               get(boost::edge_weigh_t, graph);
    //property_map<Graph, edge_weight_t>::type weightmap = get(edge_weight, g);
    //std::fprintf(stderr, "address: %p. index: %p\n", (void *)&weightmap, (void *)&index);
    for(std::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
        //auto src = source(*ei, g);
        //auto dest = target(*ei, g);
#if VERBOSE_AF
        auto v = boost::get(boost::edge_weight_t(), g, *ei);
        std::fprintf(stderr, "[%p] value: %f. s: %d. dest: %d\n", (void *)&ed, v, unsigned(index[src]), unsigned(index[dest]));
#endif
        boost::put(boost::edge_weight_t(), g, *ei, 1. / (double(std::rand()) / RAND_MAX));
#if VERBOSE_AF
        std::fprintf(stderr, "after value: %f. s: %d. dest: %d\n", boost::get(boost::edge_weight_t(), g, *ei), unsigned(index[src]), unsigned(index[dest]));
#endif
        //typename GraphTraits::edge_descriptor e;
        //std::fprintf(stderr, "edge weight: %f. edge id: %d\n", g[*ei], (int)*ei);
    }
    for(auto [vs, ve] = boost::vertices(g); vs != ve; ++vs) {
        boost::graph_traits<Graph>::vertex_descriptor vind = *vs;
        ++vind;
        //std::fprintf(stderr, "vd: %zu\n", size_t(vind));
        //auto vind = *vs;
        //TD<decltype(vind)> td;
    }
#if 0
    size_t edgecount = 0;
    for(const auto &e: g.edges()) {
        edgecount += reinterpret_cast<uint64_t>(&e) % 65536;
        //std::fprintf(stderr, "WOOO\n");
    }
    edgecount = 0;
    for(const auto &e: g.vertices()) {
        edgecount += reinterpret_cast<uint64_t>(&e) % 65536;
    }
#if VERBOSE_AF
    std::fprintf(stderr, "sum of hash remainders: %zu\n", edgecount);
#endif
#endif
    std::vector<typename Graph::Vertex> top;
    try {
        top = g.toposort();
    } catch(const boost::not_a_dag &ex) {
        std::fprintf(stderr, "Not a dag, can't topo sort\n");
    }
    return g;
}
auto csv_parse(const char *fn) {
    auto g = parse_nber<boost::undirectedS>(fn);
    using Graph = decltype(g);
    std::vector<typename Graph::Vertex> top;
    try {
        top = g.toposort();
    } catch(const boost::not_a_dag &ex) {
        std::fprintf(stderr, "Not a dag, can't topo sort\n");
    }
    using Graph = decltype(g);
    boost::graph_traits<decltype(g)>::edge_iterator ei, ei_end;
#if 0
    typedef boost::graph_traits<Graph> GraphTraits;
    GraphTraits gt;
    std::fprintf(stderr, "%p\n", (void *)&gt);
#endif
    //typename boost::property_map<Graph, boost::vertex_index_t>::type index = get(boost::vertex_index, g);
    return g;
}

int main(int c, char **v) {
    std::string input = c == 1 ? "../data/dolphins.graph": const_cast<const char *>(v[1]);
    fgc::Graph<undirectedS> g;
    if(input.find(".csv") != std::string::npos) {
        g = csv_parse(input.data());
    } else if(input.find(".gr") != std::string::npos && input.find(".graph") == std::string::npos) {
        g = dimacs_official_parse(input);
    } else {
        // DIMACS, non-official
        g = dimacs_parse(input.data());
        //if(false) {
        //}
    }
    std::vector<uint32_t> ccomp(boost::num_vertices(g));
    unsigned ncomp = boost::connected_components(g, &ccomp[0]);
    if(ncomp != 1) {
        std::fprintf(stderr, "not connected. ncomp: %u\n", ncomp);
        return 1;
    }
    uint64_t seed = 1337;
    //min(log2(n)^(2.5), 3000)
    size_t nsampled_max = std::min(std::ceil(std::pow(std::log2(boost::num_vertices(g)), 2.5)), 3000.);
    if(nsampled_max > boost::num_vertices(g))
        nsampled_max = boost::num_vertices(g) / 2;
    double frac = nsampled_max / double(boost::num_vertices(g));
    auto sampled = thorup_sample(g, 10, seed, frac); // 0 is the seed, 500 is the maximum sampled size
    std::fprintf(stderr, "sampled size: %zu\n", sampled.size());
    std::fprintf(stderr, "ncomp: %u\n", ncomp);
#if VERBOSE_AF
    for(const auto v: sampled) {
        std::vector<double> distances(boost::num_vertices(g));
        boost::dijkstra_shortest_paths(g, v, distance_map(&distances[0]));
        std::fprintf(stderr, "v %zu has max distance %f\n", size_t(v), *std::max_element(distances.begin(), distances.end()));
    }
#endif
#if 1
    auto med_solution =  fgc::jain_vazirani_kmedian(g, sampled, 10);
    for(const auto v: med_solution) assert(std::find(sampled.begin(), sampled.end(), v) != sampled.end());
    std::fprintf(stderr, "med solution size: %zu\n", med_solution.size());
    auto [costs, assignments] = get_costs(g, med_solution);
#else
    if(sampled.size() > boost::num_vertices(g) / 2)
        sampled.erase(sampled.begin() + sampled.size() / 2, sampled.end());
    auto [costs, assignments] = get_costs(g, sampled);
#endif
    coresets::CoresetSampler<float, uint32_t> sampler;
    std::fprintf(stderr, "constructed sampler\n");
    // TODO: make assignments real
#if 1
    sampler.make_sampler(costs.size(), med_solution.size(), costs.data(), assignments.data());
#else
    sampler.make_sampler(costs.size(), sampled.size(), costs.data(), assignments.data());
#endif
    std::fprintf(stderr, "made sampler\n");
    auto sampled_cs = sampler.sample(50);
}
