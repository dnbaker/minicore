#include "graph.h"
#include <string>
#include <fstream>

graph::UndirGraph<> parse_unweighted(const char *fn) {
    std::string line;
    std::ifstream ifs(fn);
    if(!std::getline(ifs, line)) throw 1;
    unsigned nnodes = std::atoi(line.data());
    if(!nnodes) throw 2;
    unsigned nedges = std::atoi(std::strchr(line.data(), ' ') + 1);
    if(!nedges) throw 2;
    graph::UndirGraph<> ret(nnodes);
    unsigned lastv = std::atoi(std::strchr(std::strchr(line.data(), ' ') + 1, ' ') + 1);
    unsigned id = 0;
    using edge_property_type = typename decltype(ret)::edge_property_type;
    while(std::getline(ifs, line)) {
        const char *s = line.data();
        if(!std::isdigit(*s)) continue;
        for(;;) {
            std::fprintf(stderr, "Adding edge from %d to %d\n", id, std::atoi(s));
            //boost::add_edge(id, std::atoi(s), ret);
            boost::add_edge(id, std::atoi(s), static_cast<edge_property_type>(1.), ret);

            if((s = std::strchr(s, ' ')) == nullptr) break;
            ++s;
        }
        ++id;
    }
    return ret;
}

using namespace graph;

int main(int c, char **v) {
    auto g = parse_unweighted(c == 1 ? "../dolphins.graph": const_cast<const char *>(v[1]));
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
}
