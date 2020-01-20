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


template<typename Graph, typename RNG, typename ICon, typename FCon, typename IT, typename RetCon>
void calculate_distortion_centerset(Graph &x, unsigned k, const ICon &indices, FCon &costbuffer,
                             const std::vector<coresets::IndexCoreset<IT, typename graph::edge_distance_type>> &coresets,
                             RetCon &ret)
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
        //OMP_PRAGMA("omp parallel for reduction(+:coreset_cost)")
        for(unsigned i = 0; i < cssz; ++i) {
            coreset_cost += costbuffer[indices[i]] * weights[i];
        }
        ret[j] = coreset_cost;
    }
    for(size_t j = 0; i < ncs; ++j) {
        double distortion = std::abs(ret[j] / fullcost - 1.);
        std::fprintf(stderr, "distortion for coreset %zu of size %zu is %g\n", j, coresets[j].size(), distortion);
        ret[j] = distortion;
    }
}

#if 0
fgc::Graph<undirectedS> &
max_component(fgc::Graph<undirectedS> &g) {
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

    fgc::Graph<undirectedS, float> g = parse_by_fn(input);
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

    // Perform Thorup sample before JV method.
    auto lsearcher = make_kmed_lsearcher(~dm, k, 1e-5, seed);
    lsearcher.run();
    auto med_solution = lsearcher.sol_;
    auto ccost = lsearcher.current_cost_;
    std::fprintf(stderr, "cost: %f\n", ccost);
#ifndef NDEBUG
    for(const auto ms: med_solution) {
        assert(ms < lsearcher.mat_.rows());
        if(ptr) assert(ms < sampled.size());
        assert(ms < boost::num_vertices(g));
    }
#endif
    // Calculate the costs of this solution
    std::vector<uint32_t> approx_v(med_solution.begin(), med_solution.end());
    if(ptr) for(auto &i: approx_v) i = sampled[i]; // Remember, these were indices into the sampled vector, not the original solution
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
    std::ofstream tblout(output_prefix + ".table_out.tsv");
    tblout << "#label" << ':' << "coreset_size" << 'x' << "sampled#times" << '\t' << "mincost" << '\t' << "meancost" << '\t' << "maxcost" << '\n';
    static constexpr unsigned nsamples = 1000;
    sample_cost_full(g, rng, tblout, k, nsamples);
    for(auto coreset_size: coreset_sizes) {
        if(auto nr = (~dm).rows(); nr < coreset_size) coreset_size = nr;
        char buf[128];
        std::sprintf(buf, ".sampled.%u.matcs", coreset_size);
        std::string fn = output_prefix + buf;
        auto sampled_cs = sampler.sample(coreset_size);
        //auto subm = submatrix(~dm, 0, 0, coreset_size, (~dm).columns());
        auto subm = blaze::DynamicMatrix<float>(coreset_size, (~dm).columns());
        std::fprintf(stderr, "About to fill distmat with coreset of size %u\n", coreset_size);
        fill_graph_distmat(g, subm, &sampled_cs.indices_);
        if(z != 1.)
            subm = blaze::pow(blaze::abs(subm), z);
        // tmpdm has # indices rows, # nodes columns
        auto columnsel = columns(subm, sampled_cs.indices_.data(), sampled_cs.indices_.size());
        blaze::DynamicMatrix<float> coreset_dm = columns(subm, sampled_cs.indices_.data(), sampled_cs.indices_.size());
        assert(coreset_dm.rows() == coreset_dm.columns());
        blaze::Archive<std::ofstream> bfp(fn);
        bfp << sampled_cs.indices_ << sampled_cs.weights_ << coreset_dm;
        for(size_t i = 0; i < coreset_dm.columns(); ++i) {
            column(coreset_dm, i) *= sampled_cs.weights_[i];
        }
        sample_and_write(coreset_dm, rng, tblout, k, std::string("graphcoreset,z="), nsamples);
    }
}
