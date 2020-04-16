/*

 Creates a DIMACS sp .gr file from an OSM file.

*/

#include <cstdlib>  // for std::exit
#include <iostream> // for std::cout, std::cerr
#include <cstdio>   // std::FILE *
#include "robin-hood-hashing/src/include/robin_hood.h"

// Allow any format of input files (XML, PBF, ...)
#include <osmium/io/any_input.hpp>

// For the osmium::geom::haversine::distance() function
#include <osmium/geom/haversine.hpp>

// For osmium::apply()
#include <osmium/visitor.hpp>

// For the location index. There are different types of indexes available.
// This will work for all input files keeping the index in memory.
#include <osmium/index/map/flex_mem.hpp>

// For the NodeLocationForWays handler
#include <osmium/handler/node_locations_for_ways.hpp>
//#include <unordered_set>
#include <vector>
#include <cinttypes>

#include "minocore/graph.h"

// The type of index used. This must match the include file above
using index_type = osmium::index::map::FlexMem<osmium::unsigned_object_id_type, osmium::Location>;

// The location handler always depends on the index type
using location_handler_type = osmium::handler::NodeLocationsForWays<index_type>;

// This handler only implements the way() function, we are not interested in
// any other objects.
using id_int_t = osmium::object_id_type;
struct OSM2DimacsHandler : public osmium::handler::Handler {
    using LatLon = std::pair<double, double>;

    //double length = 0;
    robin_hood::unordered_flat_map<id_int_t, LatLon> node_ids_;
    struct EdgeD {
        id_int_t lhs_, rhs_;
        double dist_;
    };
    std::vector<EdgeD> edges_;

    // If the way has a "highway" tag, find its length and add it to the
    // overall length.
    void way(const osmium::Way& way) {
        //const char* highway = way.tags()["highway"];
        if(way.tags()["highway"] == nullptr) return;
        //length += osmium::geom::haversine::distance(way.nodes());
        auto &nodes = way.nodes();
        size_t i = 0;
        while(!nodes[i].location()) ++i;
        size_t j = i;
        while(++j < nodes.size()) {
            if(!nodes[j].location()) continue;
            auto jloc = nodes[j].location();
            node_ids_.emplace(nodes[i].positive_ref(), std::make_pair(nodes[i].location().lat(), nodes[i].location().lon()));
            node_ids_.emplace(nodes[j].positive_ref(), std::make_pair(jloc.lat(), jloc.lon()));
            auto dist = osmium::geom::haversine::distance(nodes[i].location(), jloc);
            edges_.push_back(EdgeD({static_cast<id_int_t>(nodes[i].positive_ref()), static_cast<id_int_t>(nodes[j].positive_ref()), dist}));
            i = j;
        }
    }

}; // struct OSM2DimacsHandler

int main(int argc, char* argv[]) {
    if (argc < 2 || std::find_if(argv, argv + argc, [](auto x) {return std::strcmp(x, "-h") == 0;}) != argv + argc) {
        std::cerr << "Usage: " << argv[0] << " OSMFILE <high_freq_bbox> <output ? output: stdout>\n";
        std::cerr << "Consumes an osm file and emits a DIMACS .gr file which can then be used.\n"
                  << "which ultimately could be transformed into a graph parser, but\n"
                  << "I see no reason to not just let it be a preprocessing step\n"
                  << "Additionally annotates positions with their latitude and longitude.\n";
        std::exit(1);
    }
    std::FILE *ofp = argc < 3 ? stdout: std::fopen(argv[2], "w");

    try {
        // Initialize the reader with the filename from the command line and
        // tell it to only read nodes and ways.
        osmium::io::Reader reader{argv[1], osmium::osm_entity_bits::node | osmium::osm_entity_bits::way};

        // The index to hold node locations.
        index_type index;

        // The location handler will add the node locations to the index and then
        // to the ways
        location_handler_type location_handler{index};

        location_handler.ignore_errors();

        // Our handler defined above
        OSM2DimacsHandler road_length_handler;

        // Apply input data to first the location handler and then our own handler
        osmium::apply(reader, location_handler, road_length_handler);

        id_int_t assigned_id = 1;
        robin_hood::unordered_map<id_int_t, id_int_t> reassigner;
        reassigner.reserve(road_length_handler.node_ids_.size());
        std::fprintf(ofp, "c Auto-generated 9th DIMACS Implementation Challenge: Shortest Paths-format file\n"
                          "c From Open Street Maps [OSM] (https://openstreetmap.org)\n"
                          "c Using libosmium\n"
                          "c Following this line are node reassignments from ids to parsed node ids, all marked as comments lines.\n"
                          "p sp %zu %zu\n",
                     road_length_handler.node_ids_.size(), road_length_handler.edges_.size());
        for(const auto [id, location]: road_length_handler.node_ids_) {
            reassigner[id] = assigned_id;
            std::fprintf(ofp, "c %" PRId64 "->%" PRId64 "\t%0.12g\t%0.12g\n", id, assigned_id++, location.first, location.second);
        }
        for(const auto &edge: road_length_handler.edges_) {
            auto lhs = reassigner[edge.lhs_], rhs = reassigner[edge.rhs_];
            std::fprintf(ofp, "a %" PRId64 " %" PRId64 " %g\n", lhs, rhs, edge.dist_);
        }

        // Output the length. The haversine function calculates it in meters,
        // so we first devide by 1000 to get kilometers.
        //std::cout << "Length: " << road_length_handler.length / 1000 << " km\n";
    } catch (const std::exception& e) {
        // All exceptions used by the Osmium library derive from std::exception.
        std::cerr << e.what() << '\n';
        std::exit(1);
    }
    if(ofp != stdout) std::fclose(ofp);
}

