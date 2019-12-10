#include "graph.h"
#include "parse.h"
#include "bicriteria.h"

template<typename T> class TD;


using namespace graph;

void dimacs_parse(const char *fn) {
    auto g = parse_dimacs_unweighted<boost::undirectedS>(fn);
    using Graph = decltype(g);
    boost::graph_traits<decltype(g)>::edge_iterator ei, ei_end;
    typedef boost::graph_traits<Graph> GraphTraits;
    typename boost::property_map<Graph, boost::vertex_index_t>::type index = get(boost::vertex_index, g);
    //typename boost::property_map<Graph, boost::edge_weight_t>::type weight = get(boost::edge_weight, g);
    //typename boost::property_map<Graph, boost::edge_weight_t>::type weightMap = 
    //               get(boost::edge_weigh_t, graph);
    for(std::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
        auto ed = *ei;
        auto src = source(*ei, g);
        auto dest = target(*ei, g);
        std::fprintf(stderr, "value: %f. s: %d. dest: %d\n", g[ed], index[src], index[dest]);
        //typename GraphTraits::edge_descriptor e;
        //std::fprintf(stderr, "edge weight: %f. edge id: %d\n", g[*ei], (int)*ei);
    }
    for(auto [vs, ve] = boost::vertices(g); vs != ve; ++vs) {
        boost::graph_traits<Graph>::vertex_descriptor vind = *vs;
        std::fprintf(stderr, "vd: %zu\n", size_t(vind));
        //auto vind = *vs;
        //TD<decltype(vind)> td;
    }
    for(const auto &e: g.edges()) {
        //std::fprintf(stderr, "WOOO\n");
    }
    for(const auto &e: g.vertices()) {
        //std::fprintf(stderr, "WOOO\n");
    }
    std::vector<typename Graph::Vertex> top;
    try {
        top = g.toposort();
    } catch(const boost::not_a_dag &ex) {
        std::fprintf(stderr, "Not a dag, can't topo sort\n");
    }
}
void csv_parse(const char *fn) {
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
}

int main(int c, char **v) {
    std::string input = c == 1 ? "../dolphins.graph": const_cast<const char *>(v[1]);
    if(input.find(".csv") != /*std::string::*/input.npos) {
        csv_parse(input.data());
    } else {
        dimacs_parse(input.data());
    }
}
