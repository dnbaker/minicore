#pragma once
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/topological_sort.hpp"
#include "boost/graph/graph_traits.hpp"
#include "boost/graph/dijkstra_shortest_paths.hpp"
#include "boost/graph/connected_components.hpp"
#include "boost/property_map/property_map.hpp"
#include "shared.h"

namespace fgc {

// May replace with robin_hood if they implement this.

#if 0
template<typename WeightType>
struct EdgeWeight<WeightType>: boost::property<boost::edge_weight_t, WeightType> {
    using super = boost::property<boost::edge_weight_t, WeightType>;
    template<typename T>
    EdgeWeight(T x): boost::property<boost::edge_weight_t, WeightType>(x) {}
};
#endif
using namespace boost;

template<typename DirectedS=undirectedS, typename EdgeProps=float, typename VtxProps=boost::no_property,
         typename GraphProps=boost::no_property>
struct Graph: boost::adjacency_list<vecS, vecS, DirectedS, VtxProps, boost::property<boost::edge_weight_t, EdgeProps>, GraphProps> {
    using super = boost::adjacency_list<vecS, vecS, DirectedS, VtxProps, boost::property<boost::edge_weight_t, EdgeProps>, GraphProps>;
    using this_type = Graph<DirectedS, EdgeProps, VtxProps, GraphProps>;
    using edge_distance_type = EdgeProps;

    template<typename...Args>
    Graph(Args &&... args): super(std::forward<Args>(args)...) {
    }
    size_t num_edges() const {return boost::num_edges(*this);}
    size_t num_vertices() const {return boost::num_vertices(*this);}

    using edge_iterator         = decltype(boost::edges(std::declval<this_type>()).first);
    using edge_const_iterator   = decltype(boost::edges(std::declval<std::add_const_t<this_type>>()).first);
    using vertex_iterator       = decltype(boost::vertices(std::declval<this_type>()).first);
    using vertex_const_iterator = decltype(boost::vertices(std::declval<std::add_const_t<this_type>>()).first);
    using adjacency_iterator    = typename graph_traits<Graph>::adjacency_iterator;
    using vertex_descriptor     = typename graph_traits<Graph>::vertex_descriptor;
    using edge_descriptor       = typename graph_traits<Graph>::edge_descriptor;

    using Vertex                = vertex_descriptor;
    using Edge                  = edge_descriptor;

    static_assert(std::is_same_v<edge_iterator, edge_const_iterator>, "are they tho?");
    static_assert(std::is_same_v<vertex_iterator, vertex_const_iterator>, "are they tho?");


    vertex_descriptor add_vertex() {
        return boost::add_vertex(*this);
    }

    template<typename...Args>
    std::pair<edge_descriptor, bool> add_edge(Vertex u, Vertex v) {
        return boost::add_edge(u, v, *this);
    }
    template<typename EProps>
    std::pair<edge_descriptor, bool> add_edge(Vertex u, Vertex v, const EProps &prop) {
        return boost::add_edge(u, v, prop, *this);
    }

    struct Vertices {
        vertex_iterator f_;
        vertex_iterator e_;
        Vertices(Graph &ref) {
            std::tie(f_, e_) = boost::vertices(ref);
        }
        auto begin() const {
            return f_;
        }
        auto end() const {
            return e_;
        }
    };
    struct ConstVertices {
        vertex_const_iterator f_;
        vertex_const_iterator e_;
        ConstVertices(const Graph &ref) {
            std::tie(f_, e_) = boost::vertices(ref);
        }
        auto begin() const {
            return f_;
        }
        auto end() const {
            return e_;
        }
    };
    struct Edges {
        edge_iterator f_;
        edge_iterator e_;
        Edges(Graph &ref) {
            std::tie(f_, e_) = boost::edges(ref);
        }
        auto begin() const {
            return f_;
        }
        auto end() const {
            return e_;
        }
    };
    struct ConstEdges {
        edge_const_iterator f_;
        edge_const_iterator e_;
        ConstEdges(const Graph &ref) {
            std::tie(f_, e_) = boost::edges(ref);
        }
        auto begin() const {
            return f_;
        }
        auto end() const {
            return e_;
        }
    };
    struct Adjacencies {
        adjacency_iterator f_, e_;
        Adjacencies(Vertex vd, const Graph &ref) {
            std::tie(f_, e_) = boost::adjacent_vertices(vd, ref);
        }
        auto begin() const {return f_;}
        auto end()   const {return e_;}
    };
    auto edges() {
        return Edges(*this);
    }
    auto cedges() const {
        return ConstEdges(*this);
    }
    auto edges() const {return cedges();}
    auto vertices() {
        return Vertices(*this);
    }
    auto vertices() const {
        return cvertices();
    }
    auto cvertices() const {
        return ConstVertices(*this);
    }
    template<typename F>
    void for_each_edge(const F &f) {
        auto e = edges();
        std::for_each(e.begin(), e.end(), f);
    }
    template<typename F>
    void for_each_edge(const F &f) const {
        auto e = edges();
        std::for_each(e.begin(), e.end(), f);
    }
    template<typename F>
    void for_each_vertex(const F &f) {
        auto v = vertices();
        std::for_each(v.begin(), v.end(), f);
    }
    template<typename F>
    void for_each_vertex(const F &f) const {
        auto v = vertices();
        std::for_each(v.begin(), v.end(), f);
    }
    auto try_toposort() const {
        using Type = decltype(this->toposort());
        std::unique_ptr<Type> ret;
        try {
            ret.reset(new Type(toposort()));
        } catch(const boost::not_a_dag &ex) {
            // No sweat
        }
        return ret;
    }
    template<typename Allocator=std::allocator<Vertex>>
    auto toposort() const {
        // TODO: consider doing it as
#if 0
        std::vector<Vertex, Allocator> ret(num_vertices());
        auto rit = ret.rbegin();
        // Or
        // std::pair<std::unique_ptr<Vertex[], Deleter>, size_t> ret{static_cast<Vertex *>(std::malloc(num_vertices() * sizeof(Vertex))), num_vertices()};
        // boost::topological_sort(*this, std::reverse_iterator<Vertex *>(ret.first.get() + ret.second));
        boost::topological_sort(*this, rit);
#else
        std::vector<Vertex, Allocator> ret;
        ret.reserve(num_vertices());
        boost::topological_sort(*this, std::back_inserter(ret));
        std::reverse(ret.begin(), ret.end());
#endif
        return ret;
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

template<typename Graph>
#ifndef NDEBUG
void assert_connected(const Graph &x) {
    auto ccomp = std::make_unique<typename boost::graph_traits<Graph>::vertex_descriptor[]>(boost::num_vertices(x));
    assert(boost::connected_components(x, ccomp.get()) == 1);
}
#else
void assert_connected(const Graph &) {
}
#endif


} // graph
