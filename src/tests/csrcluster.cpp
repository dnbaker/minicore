#include "include/minicore/clustering/solve.h"
#include "blaze/util/Serialization.h"

        
namespace clust = minicore::clustering;
using namespace minicore;

#define FLOAT_TYPE float
#define OTHER_FLOAT_TYPE double

#ifdef USE_DOUBLES
#undef FLOAT_TYPE
#undef OTHER_FLOAT_TYPE
#define FLOAT_TYPE double
#define OTHER_FLOAT_TYPE float
#endif

using IndptrT = uint64_t;
using IndicesT = uint32_t;
using DataT = FLOAT_TYPE;


using minicore::util::CSparseMatrix;
using minicore::util::row;
int main(int argc, char *argv[]) {
    int NUMITER = 100;
    if(const char *s = std::getenv("NUMITER")) {
        NUMITER = std::atoi(s) > 0 ? std::atoi(s): 1;
    }
    //const bool with_replacement = std::getenv("WITHREPS");
    const bool use_simd_sample = true;
    //const bool use_importance_sampling = std::getenv("USE_IMPORTANCE_SAMPLING");
    std::srand(0);
    std::ios_base::sync_with_stdio(false);
    int nthreads = 1;
    unsigned int k = 10;
    
    //FLOAT_TYPE temp = 1.;
    dist::DissimilarityMeasure msr = dist::MKL;
    blz::DV<FLOAT_TYPE> prior{FLOAT_TYPE(1)};
    bool loaded_blaze = false;
    bool skip_empty = false, transpose = true;
    std::string outprefix;
    std::vector<double> vd{1.,2.,5.};
    std::vector<int> vi{1, 2, 5};
    auto v = util::CSparseVector<double, int>(vd.data(), vi.data(), 3, 6);
    for(const auto &pair: v) std::fprintf(stderr, "index: %u, v: %g\b", pair.index(), pair.value());
    auto vit = v.begin();
    assert(vit->value() == 1.);
    assert(vit->index() == 1);
    ++vit;
    assert(vit->value() == 2.);
    assert(vit->index() == 2);
    ++vit;
    assert(vit->value() == 5.);
    assert(vit->index() == 5);
    
    for(int c;(c = getopt(argc, argv, "o:M:z:m:p:P:k:TEh?")) >= 0;) {switch(c) {
        case 'T': transpose = false; break;
        case 'm': msr = (dist::DissimilarityMeasure)std::atoi(optarg); break;
        case 'P': prior[0] = std::atof(optarg); break;
        case 'p': nthreads = std::atoi(optarg); break;
        case 'k': k = std::atoi(optarg); break;
        case 'E': skip_empty = true; break;
        case 'o': outprefix = optarg; break;
        case '?':
        case 'h':dist::print_measures();  
                std::fprintf(stderr, "Usage: %s <flags> \n-z: load blaze matrix from path\n-P: set prior (1.)\n-T set temp [1.]\n-p set num threads\n-m Set measure (MKL, 5)\n-k: set k [10]\t-T transpose mtx file\t-M parse mtx file from argument\n", *argv);
                return EXIT_FAILURE;
    }}
    OMP_ONLY(omp_set_num_threads(nthreads);)
    std::string prefix;
    blz::DV<DataT> data;
    blz::DV<IndicesT> indices;
    blz::DV<IndptrT> indptr;

#ifndef NCOLS
#define NCOLS 100
#endif
#ifndef NROWS
#define NROWS 1000
#endif
    unsigned nrows = NROWS;
    unsigned ncols = NCOLS;
    unsigned avg_nnz = std::ceil(ncols * 0.85);
    indptr.resize(nrows + 1);
    indptr[0] = 0;
    int avg_sample = 14;
    std::poisson_distribution<IndicesT> pd(avg_nnz);
    std::poisson_distribution<IndicesT> pd2(avg_sample);
    wy::WyRand<std::conditional_t<sizeof(IndicesT) <= 4, uint32_t, uint64_t>, 4> rng;
    for(size_t i = 0; i < nrows; ++i) {
        auto nextnnz = std::min(pd(rng), IndicesT(ncols));
        indptr[i + 1] = indptr[i] + nextnnz;
    }
    unsigned total_nnz = indptr[nrows];
    data.resize(total_nnz);
    indices.resize(total_nnz);
    for(size_t i = 0; i < nrows; ++i) {
        // Select total nnz without replacement
        shared::flat_hash_set<uint32_t> used_columns;
        auto ipstart = indptr[i], ipend = indptr[i + 1];
        auto  nnzinrow = ipend - ipstart;
        while(used_columns.size() < nnzinrow) {
            used_columns.insert(std::rand() % ncols);
        }
        // Copy out and sort
        std::copy(used_columns.begin(), used_columns.end(), &indices[ipstart]);
        std::sort(&indices[ipstart], &indices[ipend]);
        for(size_t i = ipstart; i != ipend; ++i) {
            data[i] = std::max(pd2(rng), IndicesT(1));
        }
    }
    CSparseMatrix<DataT, IndicesT, IndptrT> x(data.data(), indices.data(), indptr.data(), nrows, ncols, total_nnz);
    const size_t nr = x.rows(), nc = x.columns();
    std::fprintf(stderr, "prior: %g\n", prior[0]);
    std::fprintf(stderr, "msr: %d/%s\n", (int)msr, dist::msr2str(msr));
    std::vector<blaze::CompressedVector<FLOAT_TYPE, blaze::rowVector>> centers;
    std::vector<blaze::CompressedVector<FLOAT_TYPE, blaze::rowVector>> ocenters;
    const FLOAT_TYPE psum = prior[0] * nc;
    blz::DV<FLOAT_TYPE> rowsums = util::sum<blz::rowwise>(x);
    blz::DV<FLOAT_TYPE> centersums(k);
    blz::DV<FLOAT_TYPE> hardcosts;
    blz::DV<uint32_t> asn(nr, 0);
    std::vector<uint64_t> ids;
    blz::DM<FLOAT_TYPE> complete_hardcost;
    auto fp = x.rows() <= 0xFFFFFFFFu ? size_t(std::rand() % x.rows()): size_t(((uint64_t(std::rand()) << 32) | std::rand()) % x.rows());
    ids = {fp};
    blz::DynamicVector<FLOAT_TYPE, blz::rowVector> fctr;
    {
        util::assign(fctr, row(x, fp));
        centers.emplace_back(std::move(fctr));
    }
    centersums[0] = blz::sum(centers[0]);
    hardcosts = blaze::generate(nr, [&](auto id) {
        return cmp::msr_with_prior(msr, row(x, id, blz::unchecked), centers[0], prior, psum, rowsums[id], centersums[0]);
    });
    std::vector<IndptrT> selected_points;
    while(centers.size() < k) {
        size_t index;
        if(use_simd_sample) {
            index = reservoir_simd::sample(hardcosts.data(), nr, rng());
        } else {
            blz::DV<FLOAT_TYPE> tmp(hardcosts.size());
            std::partial_sum(hardcosts.begin(), hardcosts.end(), tmp.begin());
            index = std::lower_bound(tmp.begin(), tmp.end(), std::uniform_real_distribution<FLOAT_TYPE>()(rng) * hardcosts[hardcosts.size() - 1]) - tmp.begin();
        }
        selected_points.push_back(index);
        std::fprintf(stderr, "Selected point %zu with cost %g\n", index, hardcosts[index]);
        const auto cid = centers.size();
        hardcosts[index] = 0;
        util::assign(fctr, row(x, index));
        centers.push_back(fctr);
        //std::cerr << "center" << centers.back() << ", vs fctr " << fctr << '\n';
        centersums[cid] = rowsums[index];
        OMP_PFOR
        for(size_t id = 0; id < nr; ++id) {
            if(id == index) {
                asn[id] = cid; hardcosts[id] = 0.;
            } else {
                auto v = cmp::msr_with_prior(msr, row(x, id, blz::unchecked), centers[cid], prior, psum, rowsums[id], centersums[cid]);
                if(v < hardcosts[id]) {
                    //std::fprintf(stderr, "point %u is now assigned to center %u because %g < %g\n", int(id), int(cid), v, hardcosts[id]);
                    hardcosts[id] = v, asn[id] = cid;
                } else {
                    //std::fprintf(stderr, "point %u is still assigned to center %u because %g > %g\n", int(id), asn[id], v, hardcosts[id]);
                }
            }
        }
    }
    {
        std::string outcenters = outprefix + ".centers";
        std::FILE *fp = std::fopen(outcenters.data(), "wb");
        if(!fp) throw 1;
        std::sort(selected_points.begin(), selected_points.end());
        for(const auto sp: selected_points) std::fprintf(fp, "%u\n", int(sp));
        std::fclose(fp);
    }
    complete_hardcost = blaze::generate(nr, k, [&](auto r, auto col) {
        return cmp::msr_with_prior(msr, row(x, r, blz::unchecked), centers[col], prior, psum, rowsums[r], centersums[col]);
    });
    //assert(blaze::min<blaze::rowwise>(complete_hardcost) == 
    ocenters = centers;
    assert(rowsums.size() == x.rows());
    assert(centersums.size() == centers.size());
    auto mnc = blz::min(hardcosts);
    std::fprintf(stderr, "Total cost: %g. max cost: %g. min cost: %g. mean cost:%g\n", blz::sum(hardcosts), blz::max(hardcosts), mnc, blz::mean(hardcosts));
    std::vector<uint32_t> counts(k);
    for(const auto v: asn) ++counts[v];
    for(unsigned i = 0; i < k; ++i) {
        std::fprintf(stderr, "Center %d with sum %g has %u supporting, with total cost of assigned points %g\n", i, blz::sum(centers[i]), counts[i],
                     blz::sum(blz::generate(nr, [&](auto id) { return asn[id] == i ? hardcosts[id]: 0.;})));
    }
    assert(max(asn) == centers.size() - 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    clust::perform_hard_clustering(x, msr, prior, centers, asn, hardcosts);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "Wall time for clustering: %gms\n", std::chrono::duration<FLOAT_TYPE, std::milli>(t2 - t1).count());
#if 0
    std::fprintf(stderr, "Now performing minibatch clustering\n");
    size_t mbsize = 500;
    if(char *s = std::getenv("MBSIZE")) {
        mbsize = std::strtoull(s, nullptr, 10);
    }
    std::fprintf(stderr, "mbsize: %zu\n", mbsize);
    auto mbcenters = ocenters;
    std::vector<FLOAT_TYPE> weights(x.rows(), 1.);
    std::fprintf(stderr, "minibatch clustering with no weights, %s replacement, %s importance sampling\n",  with_replacement ? "with": "without", "without");
    clust::perform_hard_minibatch_clustering(x, msr, prior, mbcenters, asn, hardcosts, (std::vector<FLOAT_TYPE> *)nullptr, mbsize, NUMITER, 10, /*reseed_after=*/10, /*with_replacement=*/with_replacement, /*seed=*/rng());
    auto mbuwcenters = ocenters;
    std::fprintf(stderr, "minibatch clustering with uniform weights, %s replacement, %s importance sampling\n",  with_replacement ? "with": "without", "without");
    clust::perform_hard_minibatch_clustering(x, msr, prior, mbuwcenters, asn, hardcosts,  &weights, mbsize, NUMITER, 10, /*reseed_after=*/10, /*with_replacement=*/with_replacement, /*seed=*/rng());
    auto is_mbcenters = ocenters;
    std::fprintf(stderr, "minibatch clustering with uniform weights, %sreplacement, %s importance sampling\n", with_replacement ? "with": "without", "without");
    clust::perform_hard_minibatch_clustering(x, msr, prior, is_mbcenters, asn, hardcosts, &weights, mbsize, NUMITER, 10, /*reseed_after=*/10, /*with_replacement=*/with_replacement, /*seed=*/rng());
#endif
}
