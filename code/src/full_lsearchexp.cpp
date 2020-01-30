#if defined(USE_BOOST_PARALLEL) && USE_BOOST_PARALLEL
#include "boost/graph/use_mpi.hpp"
#include "boost/graph/distributed/depth_first_search.hpp"
#include "fgc/relaxed_heap.hpp"
#endif
#include "fgc/graph.h"
#include "fgc/parse.h"
#include "fgc/bicriteria.h"
#include "fgc/coreset.h"
#include "fgc/lsearch.h"
#include "fgc/jv.h"
#include "fgc/timer.h"
#include <ctime>
#include <getopt.h>
#include "blaze/util/Serialization.h"

template<typename T> class TD;


using namespace fgc;
using namespace boost;



template<typename Graph, typename ICon, typename FCon, typename IT, typename RetCon, typename CSWT>
void calculate_distortion_centerset(Graph &x, const ICon &indices, FCon &costbuffer,
                             const std::vector<coresets::IndexCoreset<IT, CSWT>> &coresets,
                             RetCon &ret, double z)
{
    assert(ret.size() == coresets.size());
    const size_t nv = boost::num_vertices(x);
    const size_t ncs = coresets.size();
    {
        util::ScopedSyntheticVertex<Graph> vx(x);
        auto synthetic_vertex = vx.get();
        for(auto idx: indices)
            boost::add_edge(synthetic_vertex, idx, 0., x);
        boost::dijkstra_shortest_paths(x, synthetic_vertex, distance_map(&costbuffer[0]));
    }
    if(z != 1.) costbuffer = pow(costbuffer, z);
    double fullcost = 0.;
    OMP_PRAGMA("omp parallel for reduction(+:fullcost)")
    for(unsigned i = 0; i < nv; ++i) {
        fullcost += costbuffer[i];
    }
    const double fcinv = 1. / fullcost;
    OMP_PFOR
    for(size_t j = 0; j < ncs; ++j) {
        // In theory, this could be 
        auto &cs = coresets[j];
        const auto indices = cs.indices_.data();
        const auto weights = cs.weights_.data();
        const size_t cssz = cs.size();
        double coreset_cost = 0.;
        SK_UNROLL_8
        for(unsigned i = 0; i < cssz; ++i) {
            coreset_cost += costbuffer[indices[i]] * weights[i];
        }
        ret[j] = std::abs(coreset_cost * fcinv - 1.);
    }
}

template<typename GraphT>
GraphT &
max_component(GraphT &g) {
    auto ccomp = std::make_unique<uint32_t []>(boost::num_vertices(g));
    assert(&ccomp[0] == ccomp.get());
    unsigned ncomp = boost::connected_components(g, ccomp.get());
    if(ncomp != 1) {
        std::fprintf(stderr, "not connected. ncomp: %u\n", ncomp);
        std::vector<unsigned> counts(ncomp);
        const size_t nv = boost::num_vertices(g);
        OMP_PFOR
        for(size_t i = 0; i < nv; ++i) {
            auto ci = ccomp[i];
            OMP_ATOMIC
            ++counts[ci];
        }
        auto maxcomp = std::max_element(counts.begin(), counts.end()) - counts.begin();
        std::fprintf(stderr, "maxcmp %zu out of total %u\n", maxcomp, ncomp);
        flat_hash_map<uint64_t, uint64_t> remapper;
        size_t id = 0;
        SK_UNROLL_8
        for(size_t i = 0; i < nv; ++i) {
            if(ccomp[i] == maxcomp)
                remapper[i] = id++;
        }
        GraphT newg(counts[maxcomp]);
        typename boost::property_map <fgc::Graph<undirectedS>,
                             boost::edge_weight_t >::type EdgeWeightMap = get(boost::edge_weight, g);
        using MapIt = typename flat_hash_map<uint64_t, uint64_t>::const_iterator;
        MapIt lit, rit;
        for(const auto edge: g.edges()) {
            if((lit = remapper.find(source(edge, g))) != remapper.end() &&
               (rit = remapper.find(target(edge, g))) != remapper.end())
                boost::add_edge(lit->second, rit->second, EdgeWeightMap[edge], newg);
        }
#ifndef NDEBUG
        ncomp = boost::connected_components(newg, ccomp.get());
        assert(ncomp == 1);
#endif
        std::fprintf(stderr, "After reducing to largest connected component -- num edges: %zu. num nodes: %zu\n", newg.num_edges(), newg.num_vertices());
        std::swap(newg, g);
    }
    return g;
}

void print_header(std::ofstream &ofs, char **argv, unsigned nsamples, unsigned k, double z, size_t nv, size_t ne) {
    ofs << "##Command-line: '";
    while(*argv) {
        ofs << *argv;
        ++argv;
        if(*argv) ofs << ' ';
    }
    char buf[128];
    std::sprintf(buf, "'\n##z: %g\n##nsamples: %u\n##k: %u\n##nv: %zu\n##ne: %zu\n", z, nsamples, k, nv, ne);
    ofs << buf;
    ofs << "#coreset_size\tmax distortion (VX11)\tmean distortion (VX11)\t "
        << "max distortion (BFL16)\tmean distortion (BFL16)\t"
        << "max distortion (uniform sampling)\tmean distortion (uniform sampling)\t"
        << "mean distortion on approximate soln [VX11]\tmeandist on approx [BFL16]\tmean distortion on approximate solution, Uniform Sampling"
        << "\n";
}

void usage(const char *ex) {
    std::fprintf(stderr, "usage: %s <opts> input.gr input.coreset_sampler \n"
                         "-k\tset k [12]\n"
                         "-c\tAppend coreset size. Default: {100} (if empty)\n"
                         "-M\tSet maxmimum memory size to use. Default: 16GiB\n"
                         "-z\tset z [1.]\n",
                 ex);
    std::exit(1);
}

int main(int argc, char **argv) {
    unsigned k = 10;
    double z = 1.; // z = power of the distance norm
    std::string output_prefix;
    std::vector<unsigned> coreset_sizes;
    size_t rammax = 16uLL << 30;
    uint64_t seed = std::accumulate(argv, argv + argc, uint64_t(0),
        [](auto x, auto y) {
            return x ^ std::hash<std::string>{}(y);
        }
    );
    std::string fn = std::to_string(seed);
    for(int c;(c = getopt(argc, argv, "b:N:T:t:p:o:M:S:z:s:c:k:R:Drh?")) >= 0;) {
        switch(c) {
            case 'k': k = std::atoi(optarg); break;
            case 'z': z = std::atof(optarg); break;
            case 'M': rammax = std::strtoull(optarg, nullptr, 10); break;
            case 'p': OMP_SET_NT(std::atoi(optarg)); break;
            case 'o': output_prefix = optarg; break;
            //case 's': fn = optarg; break;
            case 'h': default: usage(argv[0]);
        }
    }
    if(output_prefix.empty())
        output_prefix = std::to_string(seed);
    if(optind + 2 != argc) usage(argv[0]);
    std::string input = const_cast<const char *>(argv[optind]);
    std::srand(seed);
    std::fprintf(stderr, "Reading from file: %s\n", input.data());

    // Parse the graph
    util::Timer timer("parse time:");
    fgc::Graph<undirectedS, float> g = parse_by_fn(input);
    timer.stop();
    timer.display();
    std::fprintf(stderr, "nv: %zu. ne: %zu\n", boost::num_vertices(g), boost::num_edges(g));
    // Select only the component with the most edges.
    timer.restart("max component:");
    max_component(g);
    timer.report();
    assert_connected(g);
    coresets::CoresetSampler<float, uint32_t> sampler;
    sampler.read(argv[optind + 1]);
    if(sampler.size() != boost::num_vertices(g)) throw std::runtime_error("Needs to match");
    // Assert that it's connected, or else the problem has infinite cost.

    std::unique_ptr<DiskMat<float>> diskmatptr;
    std::unique_ptr<blaze::DynamicMatrix<float>> rammatptr;
    const size_t ndatarows = boost::num_vertices(g);

    timer.restart("distance matrix generation:");
    using CM = blaze::CustomMatrix<float, blaze::aligned, blaze::padded, blaze::rowMajor>;
    if(ndatarows * ndatarows * sizeof(float) > rammax) {
        std::fprintf(stderr, "%zu * %zu * sizeof(float) > rammax %zu\n", ndatarows, ndatarows, rammax);
        diskmatptr.reset(new DiskMat<float>(graph2diskmat(g, fn)));
    } else {
        rammatptr.reset(new blaze::DynamicMatrix<float>(graph2rammat(g, fn)));
    }
    timer.report();
    CM dm(diskmatptr ? diskmatptr->data(): rammatptr->data(), ndatarows, ndatarows, diskmatptr ? diskmatptr->spacing(): rammatptr->spacing());
    if(z != 1.) {
        std::fprintf(stderr, "rescaling distances by the power of z\n");
        timer.restart("z rescaling");
        assert(z > 1.);
        dm = pow(abs(dm), z);
        timer.report();
    }
    std::fprintf(stderr, "[Phase 2] Distances gathered\n");

    // Perform Thorup sample before JV method.
    timer.restart("local search:");
    auto lsearcher = make_kmed_lsearcher(dm, k, 1e-3, seed * seed + seed);
    lsearcher.run();
    timer.report();
    auto med_solution = lsearcher.sol_;
    auto ccost = lsearcher.current_cost_;
    // Free memory
    if(diskmatptr) diskmatptr.reset();
    if(rammatptr) rammatptr.reset();

    std::fprintf(stderr, "[Phase 3] Local search completed. Cost for solution: %g\n", ccost);
#if 0
    std::vector<uint32_t> approx_v(med_solution.begin(), med_solution.end());
    for(auto &i: approx_v) {
        // since these solutions are indices into the subsampled set
        i = sampled[i];
    }
#endif
    return EXIT_SUCCESS;
}
