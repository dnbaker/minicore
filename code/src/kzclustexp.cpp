#include "fgc/graph.h"
#include "fgc/parse.h"
#include "fgc/bicriteria.h"
#include "fgc/coreset.h"
#include "fgc/lsearch.h"
#include "fgc/jv.h"
#include <ctime>
#include <getopt.h>
#include "blaze/util/Serialization.h"

template<typename T> class TD;


using namespace fgc;
using namespace boost;


template<typename Graph, typename ICon, typename FCon, typename IT, typename RetCon>
void calculate_distortion_centerset(Graph &x, const ICon &indices, FCon &costbuffer,
                             const std::vector<coresets::IndexCoreset<IT, typename Graph::edge_distance_type>> &coresets,
                             RetCon &ret, double z)
{
    assert(ret.size() == coresets.size());
    const size_t nv = boost::num_vertices(x);
    const size_t ncs = coresets.size();
    util::ScopedSyntheticVertex<Graph> vx(x);
    {
        auto synthetic_vertex = vx.get();
        for(auto idx: indices)
            boost::add_edge(synthetic_vertex, idx, 0., x);
        boost::dijkstra_shortest_paths(x, synthetic_vertex, distance_map(&costbuffer[0]));
    }
    if(z != 1.) {
        costbuffer = pow(costbuffer, z);
    }
    double fullcost = 0.;
    OMP_PRAGMA("omp parallel for reduction(+:fullcost)")
    for(unsigned i = 0; i < nv; ++i) {
        fullcost += costbuffer[i];
    }
    OMP_PFOR
    for(size_t j = 0; j < ncs; ++j) {
        const auto indices = coresets[j].indices_.data();
        const auto weights = coresets[j].weights_.data();
        const size_t cssz = coresets[j].size();
        double coreset_cost = 0.;
        for(unsigned i = 0; i < cssz; ++i) {
            coreset_cost += costbuffer[indices[i]] * weights[i];
        }
        ret[j] = coreset_cost;
    }
    for(size_t j = 0; j < ncs; ++j) {
        double distortion = std::abs(ret[j] / fullcost - 1.);
        //std::fprintf(stderr, "distortion for coreset %zu of size %zu is %g\n", j, coresets[j].size(), distortion);
        ret[j] = distortion;
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
        typename boost::property_map <fgc::Graph<undirectedS>,
                             boost::edge_weight_t >::type EdgeWeightMap = get(boost::edge_weight, g);
        using MapIt = typename flat_hash_map<uint64_t, uint64_t>::const_iterator;
        MapIt lit, rit;
        for(const auto edge: g.edges()) {
            if((lit = remapper.find(source(edge, g))) != remapper.end() &&
               (rit = remapper.find(target(edge, g))) != remapper.end())
                boost::add_edge(lit->second, rit->second, EdgeWeightMap[edge], newg);
        }
        ncomp = boost::connected_components(newg, ccomp.get());
        std::fprintf(stderr, "num components: %u. num edges: %zu. num nodes: %zu\n", ncomp, newg.num_edges(), newg.num_vertices());
        assert(ncomp == 1);
        std::swap(newg, g);
    }
    return g;
}

void print_header(std::ofstream &ofs, char **argv, unsigned nsamples, unsigned k, double z) {
    ofs << "##Command-line: '";
    while(*argv) {
        ofs << *argv;
        ++argv;
        if(*argv) ofs << ' ';
    }
    char buf[128];
    std::sprintf(buf, "'\n##z: %g\n##nsamples: %u\n##k: %u\n", z, nsamples, k);
    ofs << buf;
    ofs << "#coreset_size\tmax distortion (true coreset)\tmean distortion (true coreset)\tmax distortion (uniform sampling)\tmean distortion (uniform sampling)\n";
}

void usage(const char *ex) {
    std::fprintf(stderr, "usage: %s <opts> [input file or ../data/dolphins.graph]\n"
                         "-k\tset k [12]\n"
                         "-c\tAppend coreset size. Default: {100} (if empty)\n"
                         "-s\tPath to write coreset sampler to\n"
                         "-S\tSet maximum size of Thorup subsampled data. Default: infinity\n"
                         "-M\tSet maxmimum memory size to use. Default: 16GiB\n"
                         "-R\tSet random seed. Default: hash based on command-line arguments\n"
                         "-z\tset z [1.]\n"
                         "-t\tSet number of sampled centers to test [500]\n"
                         "-T\tNumber of Thorup sampling trials [15]\n",
                 ex);
    std::exit(1);
}

int main(int argc, char **argv) {
    unsigned k = 10;
    double z = 1.; // z = power of the distance norm
    std::string fn = std::string("default_scratch.") + std::to_string(std::rand()) + ".tmp";
    std::string output_prefix;
    std::vector<unsigned> coreset_sizes;
    unsigned coreset_testing_num_iter = 500;
    //size_t nsampled_max = 0;
    size_t rammax = 16uLL << 30;
    uint64_t seed = std::accumulate(argv, argv + argc, uint64_t(0),                 
        [](auto x, auto y) {                                                       
            return x ^ std::hash<std::string>{}(y);                                
        }
    );
    unsigned num_thorup_trials = 15;
    for(int c;(c = getopt(argc, argv, "T:t:p:o:M:S:z:s:c:k:R:h?")) >= 0;) {
        switch(c) {
            case 'k': k = std::atoi(optarg); break;
            case 'z': z = std::atof(optarg); break;
            case 'R': seed = std::strtoull(optarg, nullptr, 10); break;
            case 'M': rammax = std::strtoull(optarg, nullptr, 10); break;
            case 't': coreset_testing_num_iter = std::atoi(optarg); break;
            case 'T': num_thorup_trials = std::atoi(optarg); break;
            case 'p':
#ifdef _OPENMP
                omp_set_num_threads(std::atoi(optarg));
#endif
                break;
            case 'o': output_prefix = optarg; break;
            case 's': fn = optarg; break;
            case 'c': coreset_sizes.push_back(std::atoi(optarg)); break;
            case 'S': std::fprintf(stderr, "-S removed\n"); break;
            case 'h': default: usage(argv[0]);
        }
    }
    if(coreset_sizes.empty())
        coreset_sizes.push_back(100);
    if(output_prefix.empty())
        output_prefix = std::to_string(seed);
    std::string input = argc == optind ? "../data/dolphins.graph": const_cast<const char *>(argv[optind]);
    std::srand(seed);
    std::fprintf(stderr, "Reading from file: %s\n", input.data());

    // Parse the graph
    fgc::Graph<undirectedS, float> g = parse_by_fn(input);
    // Select only the component with the most edges.
    max_component(g);
    // Assert that it's connected, or else the problem has infinite cost.

    std::vector<typename boost::graph_traits<decltype(g)>::vertex_descriptor> sampled;
    //sampled = thorup_sample(g, k, seed, nsampled_max);
    sampled = thorup_sample_mincost(g, k, seed, num_thorup_trials);
    std::fprintf(stderr, "[Phase 1] Thorup sampling complete. Sampled %zu points from input graph: %zu vertices, %zu edges.\n", sampled.size(), boost::num_vertices(g), boost::num_edges(g));

    std::unique_ptr<DiskMat<float>> diskmatptr;
    std::unique_ptr<blaze::DynamicMatrix<float>> rammatptr;

    using CM = blaze::CustomMatrix<float, blaze::aligned, blaze::padded, blaze::rowMajor>;
    if(sampled.size() * sampled.size() * sizeof(float) > rammax) {
        diskmatptr.reset(new DiskMat<float>(graph2diskmat(g, fn, &sampled, true)));
    } else {
        rammatptr.reset(new blaze::DynamicMatrix<float>(graph2rammat(g, fn, &sampled, true)));
    }
    CM dm(diskmatptr ? diskmatptr->data(): rammatptr->data(), sampled.size(), sampled.size(), diskmatptr ? diskmatptr->spacing(): rammatptr->spacing());
    if(z != 1.) {
        assert(z > 1.);
        dm = pow(abs(dm), z);
    }
    std::fprintf(stderr, "[Phase 2] Distances gathered\n");

    // Perform Thorup sample before JV method.
    auto lsearcher = make_kmed_lsearcher(dm, k, 1e-5, seed);
    lsearcher.run();
    auto med_solution = lsearcher.sol_;
    auto ccost = lsearcher.current_cost_;
    if(diskmatptr) diskmatptr.reset();
    if(rammatptr) rammatptr.reset();

    std::fprintf(stderr, "[Phase 3] Local search completed. Cost for solution: %g\n", ccost);
    // Calculate the costs of this solution
    std::vector<uint32_t> approx_v(med_solution.begin(), med_solution.end());
    for(auto &i: approx_v) {
        // since these solutions are indices into the subsampled set
        i = sampled[i];
    }
    // For locality when calculating
    std::sort(approx_v.data(), approx_v.data() + approx_v.size());
    auto [costs, assignments] = get_costs(g, approx_v);
    std::fprintf(stderr, "[Phase 4] Calculated costs and assignments for all points\n");
    if(z != 1.)
        costs = blaze::pow(costs, z);
    // Build a coreset importance sampler based on it.
    coresets::CoresetSampler<float, uint32_t> sampler;
    sampler.make_sampler(costs.size(), med_solution.size(), costs.data(), assignments.data());
    std::FILE *ofp = std::fopen(fn.data(), "wb");
    sampler.write(ofp);
    std::fclose(ofp);
    seed = std::mt19937_64(seed)();
    wy::WyRand<uint32_t, 2> rng(seed);
    std::string ofname = output_prefix + ".table_out.tsv";
    std::ofstream tblout(ofname);
    print_header(tblout, argv, coreset_testing_num_iter, k, z);
    blaze::DynamicMatrix<uint32_t> random_centers(coreset_testing_num_iter, k);
    for(size_t i = 0; i < random_centers.rows(); ++i) {
        auto r = row(random_centers, i);
        wy::WyRand<uint32_t> rng(seed + i * coreset_testing_num_iter);
        flat_hash_set<uint32_t> centers; centers.reserve(k);
        while(centers.size() < k) {
            centers.insert(rng() % boost::num_vertices(g));
        }
        auto it = centers.begin();
        for(size_t j = 0; j < r.size(); ++j)
            r[j] = *it++;
        std::sort(r.begin(), r.end());
    }
    coresets::UniformSampler<float, uint32_t> uniform_sampler(costs.size());
    std::vector<coresets::IndexCoreset<uint32_t, float>> coresets;
    // The first half of these are true coresets, the others are uniformly subsampled.
    const size_t ncs = coreset_sizes.size();
    coresets.reserve(ncs * 2);
    for(auto coreset_size: coreset_sizes) {
        coresets.emplace_back(sampler.sample(coreset_size));
    }
    for(auto coreset_size: coreset_sizes) {
        coresets.emplace_back(uniform_sampler.sample(coreset_size));
    }
    std::fprintf(stderr, "[Phase 5] Generated coresets\n");
    blaze::DynamicVector<double> maxdistortion(coresets.size(), std::numeric_limits<double>::min()),
                                 distbuffer(boost::num_vertices(g)),
                                 currentdistortion(coresets.size()),
                                 mindistortion(coresets.size(), std::numeric_limits<double>::max()),
                                 meandistortion(coresets.size(), 0.);
    for(size_t i = 0; i < random_centers.rows(); ++i) {
        //if(i % 10 == 0)
        //    std::fprintf(stderr, "Calculating distortion %zu/%zu\n", i, random_centers.rows());
        auto rc = row(random_centers, i);
        assert(rc.size() == k);
        calculate_distortion_centerset(g, rc, distbuffer, coresets, currentdistortion, z);
        maxdistortion = max(maxdistortion, currentdistortion);
        mindistortion = min(mindistortion, currentdistortion);
        meandistortion = meandistortion + currentdistortion;
    }
    meandistortion /= random_centers.rows();
    for(size_t i = 0; i < ncs; ++i) {
        tblout << coreset_sizes[i] << '\t'
               << maxdistortion[i] << '\t' << meandistortion[i] << '\t'
               << maxdistortion[i + ncs] << '\t' << meandistortion[i + ncs]
               << '\n';
    }
    std::cerr << "mean\n" << meandistortion;
    std::cerr << "max\n" << maxdistortion;
    std::cerr << "min\n" << mindistortion;
}
