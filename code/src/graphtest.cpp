#include "graph.h"
#include "parse.h"
#include "bicriteria.h"

template<typename T> class TD;


#define undirectedS bidirectionalS
using namespace og;
using namespace boost;

auto dimacs_parse(const char *fn) {
    auto g = parse_dimacs_unweighted<boost::undirectedS>(fn);
    using Graph = decltype(g);
    boost::graph_traits<decltype(g)>::edge_iterator ei, ei_end;
    //typedef boost::graph_traits<Graph> GraphTraits;
    typename boost::property_map<Graph, boost::vertex_index_t>::type index = get(boost::vertex_index, g);
    //typename boost::property_map<Graph, boost::edge_weight_t>::type weight = get(boost::edge_weight, g);
    //typename boost::property_map<Graph, boost::edge_weight_t>::type weightMap = 
    //               get(boost::edge_weigh_t, graph);
    property_map<Graph, edge_weight_t>::type weightmap = get(edge_weight, g);
    std::fprintf(stderr, "address: %p. index: %p\n", (void *)&weightmap, (void *)&index);
    for(std::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
        auto ed = *ei;
        auto src = source(*ei, g);
        auto dest = target(*ei, g);
        auto v = boost::get(boost::edge_weight_t(), g, *ei);
        std::fprintf(stderr, "[%p] value: %f. s: %d. dest: %d\n", (void *)&ed, v, unsigned(index[src]), unsigned(index[dest]));
        boost::put(boost::edge_weight_t(), g, *ei, 1.37);
        std::fprintf(stderr, "after value: %f. s: %d. dest: %d\n", boost::get(boost::edge_weight_t(), g, *ei), unsigned(index[src]), unsigned(index[dest]));
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
    size_t edgecount = 0;
    for(const auto &e: g.edges()) {
        edgecount += reinterpret_cast<uint64_t>(&e) % 65536;
        //std::fprintf(stderr, "WOOO\n");
    }
    edgecount = 0;
    for(const auto &e: g.vertices()) {
        edgecount += reinterpret_cast<uint64_t>(&e) % 65536;
    }
    std::fprintf(stderr, "sum of hash remainders: %zu\n", edgecount);
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
    typedef boost::graph_traits<Graph> GraphTraits;
    typename boost::property_map<Graph, boost::vertex_index_t>::type index = get(boost::vertex_index, g);
    return g;
}

int main(int c, char **v) {
    std::string input = c == 1 ? "../dolphins.graph": const_cast<const char *>(v[1]);
    if(input.find(".csv") != /*std::string::*/input.npos) {
        auto g = csv_parse(input.data());
        auto sampled = thorup_sample(g, 3, 0);
        std::fprintf(stderr, "sampled size: %zu\n", sampled.size());
    } else if(input.find(".gr") != input.npos && input.find(".graph") == input.npos) {
        std::ifstream ifs(input);
        og::Graph<undirectedS> dimacs_official;
        std::string graphtype;
        size_t nnodes = 0, nedges = 0;
        for(std::string line; std::getline(ifs, line);) {
            const char *p = line.data();
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
                        boost::add_vertex(dimacs_official); // Add all the vertices
                    if((p2 = std::strchr(p, ' ')) == nullptr) throw 1;
                    p = p2 + 1;
                    nedges = std::strtoull(p, nullptr, 10);
                }
                break;
                case 'a': {
                    assert(nnodes);
                    char *strend;
                    const char *p = line.data() + 2;
                    size_t lhs = std::strtoull(p, &strend, 10);
                    p = strend + 1;
                    size_t rhs = std::strtoull(p, &strend, 10);
                    p = strend + 1;
                    double dist = std::atof(p);
                    boost::add_edge(lhs, rhs, dist, dimacs_official);
                }
                default: std::fprintf(stderr, "Unexpected: this line! (%s)\n", line.data()); throw 1;
            }
        }
    } else {
        // DIMACS, non-official
        auto g = dimacs_parse(input.data());
        //if(false) {
        auto sampled = thorup_sample(g, 3, 0);
        std::fprintf(stderr, "sampled size: %zu\n", sampled.size());
        //}
    }
}
