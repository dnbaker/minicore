#pragma once
#include "boost/graph/adjacency_list.hpp"

namespace graph {
using boost::vecS;
using boost::undirectedS;
using boost::directedS;
using boost::bidirectionalS;
using boost::vertex_index;
using boost::vertex_index_t;

template<typename DirectedS=undirectedS, typename EdgeProps=float, typename VtxProps=boost::no_property,
         typename GraphProps=boost::no_property>
struct Graph: boost::adjacency_list<vecS, vecS, DirectedS, VtxProps, EdgeProps, GraphProps> {
    using super = boost::adjacency_list<vecS, vecS, DirectedS, VtxProps, EdgeProps, GraphProps>;
    template<typename...Args>
    Graph(Args &&... args): super(std::forward<Args>(args)...) {
    }
};
template<typename EdgeProps=float, typename VtxProps=boost::no_property,
         typename GraphProps=boost::no_property>
struct DirGraph: public Graph<directedS, EdgeProps, VtxProps, GraphProps> {
    using super = Graph<directedS, EdgeProps, VtxProps, GraphProps>;
    template<typename...Args>
    DirGraph(Args &&... args): super(std::forward<Args>(args)...) {
    }
};
template<typename EdgeProps=float, typename VtxProps=boost::no_property,
         typename GraphProps=boost::no_property>
struct UndirGraph: public Graph<undirectedS, EdgeProps, VtxProps, GraphProps> {
    using super = Graph<undirectedS, EdgeProps, VtxProps, GraphProps>;
    template<typename...Args>
    UndirGraph(Args &&... args): super(std::forward<Args>(args)...) {
    }
};
template<typename EdgeProps=float, typename VtxProps=boost::no_property,
         typename GraphProps=boost::no_property>
struct BidirGraph: public Graph<bidirectionalS, EdgeProps, VtxProps, GraphProps> {
    using super = Graph<bidirectionalS, EdgeProps, VtxProps, GraphProps>;
    template<typename...Args>
    BidirGraph(Args &&... args): super(std::forward<Args>(args)...) {
    }
};

} // graph
