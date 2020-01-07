#pragma once
#include "graph.h"
#include <fstream>
#include <string>
#include <climits>
#include <unordered_map>
#include <iostream>
#include <cstring>

namespace fgc {

enum ParsedFiletype {
    DIMACS,
    DIMACS_UNWEIGHTED,
    TSP,
    NBER
};

template<typename DirectedS, typename VtxProps=boost::no_property, typename GraphProps=boost::no_property>
Graph<DirectedS, float, VtxProps, GraphProps> parse_dimacs_unweighted(const char *fn) {
    using GraphType = Graph<DirectedS, float, VtxProps, GraphProps>;
    std::string line;
    std::ifstream ifs(fn);
    if(!std::getline(ifs, line)) throw std::runtime_error(std::string("Failed to read from file ") + fn);
    unsigned nnodes = std::atoi(line.data());
    if(!nnodes) throw 2;
    auto p = std::strchr(line.data(), ' ') + 1;
    unsigned nedges = std::atoi(p);
    if(!nedges) throw 2;
    GraphType ret(nnodes);
    unsigned lastv = std::atoi(std::strchr(p + 1, ' '));
    unsigned id = 0;
    using edge_property_type = typename decltype(ret)::edge_property_type;
    while(std::getline(ifs, line)) {
        const char *s = line.data();
        if(!std::isdigit(*s)) continue;
        //boost::add_edge(id, id, 0., ret);
        //std::fprintf(stderr, "added edge from %zu to %zu\n", size_t(id), size_t(id));
        for(;;) {
            auto newv = std::atoi(s) - 1;
            boost::add_edge(id, newv, static_cast<edge_property_type>(1.), ret);

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
Graph<DirectedS, float, VtxProps, GraphProps> parse_nber(const char *fn) {
    using GraphType = Graph<DirectedS, float, VtxProps, GraphProps>;
    GraphType ret;
    std::string line;
    std::ifstream ifs(fn);
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
    }
    std::fprintf(stderr, "num edges: %zu. num vertices: %zu\n", boost::num_edges(ret), boost::num_vertices(ret));
    return ret;
}

} // graph
