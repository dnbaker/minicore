#pragma once
#include "graph.h"
#include <fstream>
#include <string>
#include <climits>
#include <cassert>
#include <unordered_map>
#include <iostream>
#include <cstring>
#include "minicore/util/io.h"

namespace minicore {
using namespace ::minicore::shared;

//#define undirectedS bidirectionalS

enum ParsedFiletype {
    DIMACS,
    DIMACS_UNWEIGHTED,
    TSP,
    NBER
};
namespace graph {

template<typename DirectedS, typename VtxProps=boost::no_property, typename GraphProps=boost::no_property>
Graph<DirectedS, float, VtxProps, GraphProps> parse_dimacs_unweighted(std::string fn) {
    using GraphType = Graph<DirectedS, float, VtxProps, GraphProps>;
    std::string line;
    auto fdat = util::io::xopen(fn);
    auto &ifs = *fdat.first;
    if(!std::getline(ifs, line)) throw std::runtime_error(std::string("Failed to read from file ") + fn);
    unsigned nnodes = std::atoi(line.data());
    if(!nnodes) throw 2;
    auto p = std::strchr(line.data(), ' ') + 1;
    unsigned nedges = std::atoi(p);
    if(!nedges) throw 2;
    GraphType ret(nnodes);
    unsigned id = 0;
    using edge_property_type = typename decltype(ret)::edge_property_type;
    while(std::getline(ifs, line)) {
        const char *s = line.data();
        if(!std::isdigit(*s)) continue;
        for(;;) {
            auto newv = std::atoi(s) - 1;
            boost::add_edge(id, newv, static_cast<edge_property_type>(1.), ret);
            assert(unsigned(newv) < boost::num_vertices(ret));
            assert(id < boost::num_vertices(ret));
            if((s = std::strchr(s, ' ')) == nullptr || !std::isdigit(*++s)) break;
        }
        ++id;
    }
    std::cout << line << '\n';
    std::fprintf(stderr, "num edges: %zu. num vertices: %zu\n", boost::num_edges(ret), boost::num_vertices(ret));
    return ret;
}

// #state1,place1,mi_to_place,state2,place2
template<typename DirectedS, typename VtxProps=boost::no_property, typename GraphProps=boost::no_property, typename VtxIdType=uint64_t>
Graph<DirectedS, float, VtxProps, GraphProps> parse_nber(std::string fn) {
    using GraphType = Graph<DirectedS, float, VtxProps, GraphProps>;
    GraphType ret;
    std::string line;
    auto fdat = util::io::xopen(fn);
    auto &ifs = *fdat.first;
    std::vector<VtxIdType> ids;
    std::unordered_map<VtxIdType, uint32_t> loc2id;
    static constexpr unsigned SHIFT = sizeof(VtxIdType) * CHAR_BIT / 2;
    using edge_property_type = typename GraphType::edge_property_type;
    while(std::getline(ifs, line)) {
        if(line.empty() || line.front() == '#' || line.front() == '\n') continue;
        line.erase(std::remove(line.begin(), line.end(), '"'), line.end());
        const char *s = line.data();
        VtxIdType val = VtxIdType(std::atoi(s)) << SHIFT;
        if((s = std::strchr(s, ',')) == nullptr) throw std::runtime_error(std::string("Failed to parse from fn") + fn);
        val |= std::atoi(++s);
        auto it = loc2id.find(val);
        if(it == loc2id.end()) {
            it = loc2id.emplace(val, loc2id.size()).first;
            //auto vid = boost::add_vertex(ret);
            ids.push_back(val);
            assert(loc2id[val] == ids.size() - 1);
            assert(ids.back() == val);
            //assert(vid == ids.back());
        }
        s = std::strchr(s, ',') + 1;
        double dist = std::atof(s);
        s = std::strchr(s, ',') + 1;
        VtxIdType endval = (VtxIdType(std::atoi(s)) << SHIFT);
        endval |= std::atoi(std::strchr(s, ',') + 1);
        auto rit = loc2id.find(endval);
        if(rit == loc2id.end()) {
            rit = loc2id.emplace(endval, loc2id.size()).first;
            ids.push_back(endval);
            //auto vid = boost::add_vertex(ret);
            assert(loc2id[endval] == ids.size() - 1);
            assert(ids.back() == endval);
            //assert(vid == ids.back());
        }
        boost::add_edge(it->second, rit->second, static_cast<edge_property_type>(dist), ret);
        assert(it->second < boost::num_vertices(ret));
        assert(it->first < boost::num_vertices(ret));
    }
    std::fprintf(stderr, "num edges: %zu. num vertices: %zu\n", boost::num_edges(ret), boost::num_vertices(ret));
    return ret;
}
static minicore::Graph<undirectedS> dimacs_official_parse(std::string input) {
    minicore::Graph<undirectedS> g;
    auto fdat = util::io::xopen(input);
    std::string graphtype;
    size_t nnodes = 0, nedges = 0;
    for(std::string line; std::getline(*fdat.first, line);) {
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
                assert(lhs >= 1 || !std::fprintf(stderr, "p: %s\n", p));
                p = strend + 1;
                size_t rhs = std::strtoull(p, &strend, 10);
                assert(rhs >= 1 || !std::fprintf(stderr, "p: %s\n", p));
                p = strend + 1;
                double dist = std::atof(p);
                boost::add_edge(lhs - 1, rhs - 1, dist, g);
                assert(lhs - 1 < boost::num_vertices(g));
                assert(rhs - 1 < boost::num_vertices(g));
                break;
            }
            default: std::fprintf(stderr, "Unexpected: this line! (%s)\n", line.data()); throw std::runtime_error("");
        }
    }
    return g;
}

static minicore::Graph<undirectedS> dimacs_parse(std::string fn) {
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

static auto csv_parse(std::string fn) {
    return parse_nber<boost::undirectedS>(fn);
}

static minicore::Graph<undirectedS> parse_by_fn(std::string input) {
    minicore::Graph<undirectedS> g;
    if(input.find(".csv") != std::string::npos) {
        g = csv_parse(input);
    } else if(input.find(".gr") != std::string::npos && input.find(".graph") == std::string::npos) {
        g = dimacs_official_parse(input);
    } else g = dimacs_parse(input);
    return g;
}

} // namespace graph
using graph::parse_dimacs_unweighted;
using graph::parse_by_fn;
using graph::csv_parse;
using graph::parse_nber;
using graph::dimacs_parse;
using graph::dimacs_official_parse;


} // minicore
