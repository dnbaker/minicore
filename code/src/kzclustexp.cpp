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
    ofs << "#label\tcoreset_size\tmincost\tmeancost\tmaxcost\n";
}

void usage(const char *ex) {
    std::fprintf(stderr, "usage: %s <opts> [input file or ../data/dolphins.graph]\n"
                         "-k\tset k [12]\n"
                         "-c\tAppend coreset size. Default: {100} (if empty)\n"
                         "-s\tPath to write coreset sampler to\n"
                         "-S\tSet maximum size of Thorup subsampled data. Default: infinity\n"
                         "-M\tSet maxmimum memory size to use. Default: 16GiB\n"
                         "-R\tSet random seed. Default: hash based on command-line arguments\n"
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
    ofs << label << '\t' << mat.rows() << '\t' << mincost << '\t' << meancost << '\t' << maxcost << '\n';
}

int main(int argc, char **argv) {
    unsigned k = 10;
    double z = 1.; // z = power of the distance norm
    std::string fn = std::string("default_scratch.") + std::to_string(std::rand()) + ".tmp";
    std::string output_prefix;
    std::vector<unsigned> coreset_sizes;
    size_t nsampled_max = 0;
    size_t rammax = 16uLL << 30;
    uint64_t seed = std::accumulate(argv, argv + argc, uint64_t(0),                 
        [](auto x, auto y) {                                                       
            return x ^ std::hash<std::string>{}(y);                                
        }
    );
    for(int c;(c = getopt(argc, argv, "o:M:S:z:s:c:k:R:h?")) >= 0;) {
        switch(c) {
            case 'k': k = std::atoi(optarg); break;
            case 'z': z = std::atof(optarg); break;
            case 'R': seed = std::strtoull(optarg, nullptr, 10); break;
            case 'M': rammax = std::strtoull(optarg, nullptr, 10); break;
            case 'o': output_prefix = optarg; break;
            case 's': fn = optarg; break;
            case 'c': coreset_sizes.push_back(std::atoi(optarg)); std::fprintf(stderr, "WOOOOOOOOOOO cs back: %u\n", coreset_sizes.back()); break;
            case 'S': nsampled_max = std::strtoull(optarg, nullptr, 10); break;
            case 'h': default: usage(argv[0]);
        }
    }
    if(coreset_sizes.empty())
        coreset_sizes.push_back(100);
    if(output_prefix.empty())
        output_prefix = std::to_string(seed);
    std::string input = argc == optind ? "../data/dolphins.graph": const_cast<const char *>(argv[optind]);
    std::srand(std::hash<std::string>{}(input));

    fgc::Graph<undirectedS, float> g = parse_by_fn(input);
    max_component(g);
    // Assert that it's connected, or else the problem has infinite cost.

    std::vector<typename boost::graph_traits<decltype(g)>::vertex_descriptor> sampled;
    sampled = thorup_sample(g, k, seed, nsampled_max);
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
    if(z != 1.)
        costs = blaze::pow(blaze::abs(costs), z);
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
    static constexpr unsigned nsamples = 1000;
    print_header(tblout, argv, nsamples, k, z);
    sample_cost_full(g, rng, tblout, k, nsamples);
    for(auto coreset_size: coreset_sizes) {
        auto sampled_cs = sampler.sample(coreset_size);
        PolymorphicMat<float> coreset_mat(coreset_size, coreset_size, rammax);
        auto &coreset_dm(~coreset_mat);
        fill_graph_distmat(g, coreset_dm, &sampled_cs.indices_, true);
        // tmpdm has # indices rows, # nodes columns
        assert(coreset_dm.rows() == coreset_dm.columns());
        std::string fn(output_prefix + ".sampled." + std::to_string(coreset_size) + ".matcs");
        blaze::Archive<std::ofstream> bfp(fn);
        bfp << sampled_cs.indices_ << sampled_cs.weights_;
        for(size_t i = 0; i < coreset_dm.columns(); ++i)
            column(coreset_dm, i) *= sampled_cs.weights_[i];
        sample_and_write(coreset_dm, rng, tblout, k, "coreset", nsamples);
    }
}