#include "minocore/graph.h"
#include "minocore/parse.h"
#include "minocore/bicriteria.h"
#include "minocore/coreset.h"
#include "minocore/lsearch.h"
#include "minocore/jv.h"
#include <ctime>
#include <getopt.h>
#include "blaze/util/Serialization.h"

template<typename T> class TD;


using namespace minocore;
using namespace boost;



#if 0
minocore::Graph<undirectedS> &
max_component(minocore::Graph<undirectedS> &g) {
#endif
template<typename GraphT>
GraphT &
max_component(GraphT &g) {
    auto ccomp = std::make_unique<uint32_t []>(boost::num_vertices(g));
    assert(&ccomp[0] == ccomp.get());
    unsigned ncomp = boost::connected_components(g, ccomp.get());
    if(ncomp != 1) {
        std::fprintf(stderr, "not connected. ncomp: %u\n", ncomp);
        std::vector<unsigned> counts(ncomp);
        for(size_t i = 0, e = boost::num_vertices(g); i < e; ++counts[ccomp[i++]]);
        auto maxcomp = std::max_element(counts.begin(), counts.end()) - counts.begin();
        std::fprintf(stderr, "maxcmp %zu out of total %u\n", maxcomp, ncomp);
        flat_hash_map<uint64_t, uint64_t> remapper;
        size_t id = 0;
        for(size_t i = 0; i < boost::num_vertices(g); ++i) {
            if(ccomp[i] == maxcomp) {
                remapper[i] = id++;
            }
        }
        GraphT newg(counts[maxcomp]);
        typename boost::property_map <minocore::Graph<undirectedS>,
                             boost::edge_weight_t >::type EdgeWeightMap = get(boost::edge_weight, g);
        for(auto edge: g.edges()) {
            auto lhs = source(edge, g);
            auto rhs = target(edge, g);
            if(auto lit = remapper.find(lhs), rit = remapper.find(rhs);
               lit != remapper.end() && rit != remapper.end()) {
                boost::add_edge(lit->second, rit->second, EdgeWeightMap[edge], newg);
            }
        }
        ncomp = boost::connected_components(newg, ccomp.get());
        std::fprintf(stderr, "num components: %u. num edges: %zu. num nodes: %zu\n", ncomp, newg.num_edges(), newg.num_vertices());
        assert(ncomp == 1);
        std::swap(newg, g);
    }
    return g;
}

void usage(const char *ex) {
    std::fprintf(stderr, "usage: %s <opts> [input file or ../data/dolphins.graph]\n"
                         "-k\tset k [12]\n"
                         "-c\tAppend coreset size. Default: {100} (if empty)\n"
                         "-s\tPath to write coreset sampler to\n"
                         "-z\tset z [1.]\n",
                 ex);
    std::exit(1);
}


template<typename Mat, typename RNG>
void sample_and_write(const Mat &mat, RNG &rng, std::ofstream &ofs, unsigned k, std::string label, unsigned nsamples=1000) {
    double maxcost = 0., meancost = 0., mincost = std::numeric_limits<double>::max();
    std::vector<uint32_t> indices;
    const size_t nsamp = mat.rows();
    indices.reserve(k);
    for(unsigned i = 0; i < nsamples; ++i) {
        while(indices.size() < k)
            if(auto v = rng() % nsamp; std::find(indices.begin(), indices.end(), v) == indices.end())
                indices.push_back(v);
        double cost = blaze::sum(blaze::min<blaze::columnwise>(rows(mat, indices.data(), indices.size())));
        maxcost = std::max(cost, maxcost);
        mincost = std::min(cost, mincost);
        meancost += cost;
        indices.clear();
    }
    meancost /= nsamples;
    ofs << label << ':' << mat.rows() << 'x' << nsamples << '\t' << mincost << '\t' << meancost << '\t' << maxcost << '\n';
}

int main(int argc, char **argv) {
    unsigned k = 10;
    double z = 1.; // z = power of the distance norm
    std::string fn = std::string("default_scratch.") + std::to_string(std::rand()) + ".tmp";
    std::string output_prefix;
    std::vector<unsigned> coreset_sizes;
    size_t nsampled_max = 0;
    uint64_t hv = 0;
    for(auto av = argv; *av; hv ^= std::hash<std::string>{}(*av++));
    for(int c;(c = getopt(argc, argv, "o:S:z:s:c:k:h?")) >= 0;) {
        switch(c) {
            case 'k': k = std::atoi(optarg); break;
            case 'z': z = std::atof(optarg); break;
            case 's': fn = optarg; break;
            case 'o': output_prefix = optarg; break;
            case 'c': coreset_sizes.push_back(std::atoi(optarg)); break;
            case 'S': nsampled_max = std::strtoull(optarg, nullptr, 10); break;
            case 'h': default: usage(argv[0]);
        }
    }
    if(coreset_sizes.empty())
        coreset_sizes.push_back(100);
    if(output_prefix.empty())
        output_prefix = std::to_string(std::accumulate(argv, argv + argc, uint64_t(0),
                            [](auto x, auto y) {
                                return x ^ std::hash<std::string>{}(y);
                            }
                        ));
    std::string input = argc == optind ? "../data/dolphins.graph": const_cast<const char *>(argv[optind]);

    std::srand(std::hash<std::string>{}(input));

    minocore::Graph<undirectedS, float> g = parse_by_fn(input);
    max_component(g);
    // Assert that it's connected, or else the problem has infinite cost.
    uint64_t seed = 1337;

    if(nsampled_max == 0)
        nsampled_max = std::ceil(std::pow(std::log2(boost::num_vertices(g)), 3.5));
    std::vector<typename boost::graph_traits<decltype(g)>::vertex_descriptor> sampled;
    std::vector<typename boost::graph_traits<decltype(g)>::vertex_descriptor> *ptr = nullptr;
    if(boost::num_vertices(g) > 20000) {
        std::fprintf(stderr, "num vtx: %zu. Thorup sampling!\n", boost::num_vertices(g));
        sampled = thorup_sample(g, k, seed, nsampled_max);
        ptr = &sampled;
    }
    std::fprintf(stderr, "Thorup sampling complete\n");
    auto dm = graph2diskmat(g, fn, ptr);
    if(z != 1.) {
        assert(z > 1.);
        ~dm = pow(abs(~dm), z);
    }
}
