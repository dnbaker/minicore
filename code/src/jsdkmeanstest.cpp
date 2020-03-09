#include "fgc/jsd.h"
#include "fgc/csc.h"
#include "fgc/timer.h"

#define FT double

#define NOTDENSE 1
int main(int argc, char *argv[]) {
    if(std::find_if(argv, argv + argc, [](auto x) {return std::strcmp(x, "-h") == 0 || std::strcmp(x, "--help") == 0;})
       != argv + argc) {
        std::fprintf(stderr, "Usage: %s <max rows[1000]> <mincount[50]> <k[5]> <m[5000]>\n", *argv);
        std::exit(1);
    }
    unsigned maxnrows = argc == 1 ? 50000: std::atoi(argv[1]);
    unsigned mincount = argc <= 2 ? 50: std::atoi(argv[2]);
    unsigned k        = argc <= 3 ? 50: std::atoi(argv[3]);
    unsigned m        = argc <= 4 ? 5000: std::atoi(argv[4]);
    std::ofstream ofs("output.txt");
    auto sparsemat = fgc::csc2sparse<FT>("", true);
    std::vector<unsigned> nonemptyrows;
    size_t i = 0;
    while(nonemptyrows.size() < maxnrows && i < sparsemat.rows()) {
        const auto nzc = blz::nonZeros(row(sparsemat, i));
        //std::fprintf(stderr, "nzc: %zu\n", nzc);
        if(nzc > mincount) {
            //std::fprintf(stderr, "nzc: %zu vs min %u\n", nzc, mincount);
            nonemptyrows.push_back(i);
        }
        ++i;
        //std::fprintf(stderr, "sparsemat rows: %zu. current i: %zu\n", sparsemat.rows(), i);
    }
    std::fprintf(stderr, "Gathered %zu rows\n", nonemptyrows.size());
#ifndef NOTDENSE
    blaze::DynamicMatrix<typename decltype(sparsemat)::ElementType> filtered_sparsemat = rows(sparsemat, nonemptyrows.data(), nonemptyrows.size());
#else
    decltype(sparsemat) filtered_sparsemat = rows(sparsemat, nonemptyrows.data(), nonemptyrows.size());
#endif
    auto full_jsm = fgc::jsd::make_jsm_applicator(filtered_sparsemat, /*prior=*/blz::distance::NONE);
    fgc::util::Timer timer("k-means");
    auto kmppdat = fgc::jsd::make_kmeanspp(full_jsm, k);
    timer.report();
    std::fprintf(stderr, "finished kmpp. Now getting cost\n");
    std::fprintf(stderr, "kmpp solution cost: %g\n", blz::sum(std::get<2>(kmppdat)));
    timer.restart("kmc");
    auto kmcdat = fgc::jsd::make_kmc2(full_jsm, k, m);
    timer.report();
    std::fprintf(stderr, "finished kmc2\n");
    timer.reset();
    auto kmc2cost = fgc::coresets::get_oracle_costs(full_jsm, full_jsm.size(), kmcdat);
    std::fprintf(stderr, "kmc2 solution cost: %g\n", blz::sum(kmc2cost.second));
    std::fprintf(stderr, "\n\nNumber of cells: %zu\n", nonemptyrows.size());
    auto &[centeridx, asn, costs] = kmppdat;
    std::vector<float> counts(centeridx.size());
    blaze::DynamicMatrix<typename decltype(filtered_sparsemat)::ElementType> centers(rows(filtered_sparsemat, centeridx.data(), centeridx.size()));
    auto oracle = [](const auto &x, const auto &y) {
        return std::sqrt(fgc::jsd::multinomial_jsd(x, y));
    };
    std::fprintf(stderr, "About to start ll\n");
#if 0
    fgc::coresets::lloyd_loop(asn, counts, centers, filtered_sparsemat, 0., 50, oracle);
    std::fprintf(stderr, "About to make probdiv appl\n");
    filtered_sparsemat = rows(sparsemat, nonemptyrows.data(), nonemptyrows.size());
    auto full_jsd = fgc::jsd::make_probdiv_applicator(filtered_sparsemat, fgc::jsd::MKL, blz::distance::NONE);
    std::tie(centeridx, asn, costs) = fgc::jsd::make_kmeanspp(full_jsd, k);
    centers = rows(filtered_sparsemat, centeridx.data(), centeridx.size());
    fgc::coresets::lloyd_loop(asn, counts, centers, filtered_sparsemat, 1e-6, 250, [](const auto &x, const auto &y) {return fgc::jsd::multinomial_jsd(x, y);});
    auto coreset_sampler = fgc::jsd::make_d2_coreset_sampler(full_jsm, k, 13);
double mb_lloyd_loop(std::vector<IT> &assignments, std::vector<WFT> &counts,
                     CMatrixType &centers, MatrixType &data,
                     unsigned batch_size,
                     size_t maxiter=10000,
                     const Functor &func=Functor(),
                     uint64_t seed=137,
                     const WFT *weights=nullptr)
{
#else
    fgc::coresets::lloyd_loop(asn, counts, centers, filtered_sparsemat, 1e-6, 50, oracle);
    auto cpyasn = asn;
    auto cpycounts = counts;
    auto cpycenters = centers;
    std::mt19937_64 mt(1337);
    fgc::coresets::mb_lloyd_loop(cpyasn, cpycounts, cpycenters, filtered_sparsemat, 500, 50, oracle, mt());
    std::fprintf(stderr, "About to make probdiv appl\n");
    auto full_jsd = fgc::jsd::make_probdiv_applicator(filtered_sparsemat, fgc::jsd::MKL);
    std::tie(centeridx, asn, costs) = fgc::jsd::make_kmeanspp(full_jsd, k);
    centers = rows(filtered_sparsemat, centeridx.data(), centeridx.size());
    fgc::coresets::lloyd_loop(asn, counts, centers, filtered_sparsemat, 1e-6, 250, [](const auto &x, const auto &y) {return fgc::jsd::multinomial_jsd(x, y);});
    //auto coreset_sampler = fgc::jsd::make_d2_coreset_sampler(full_jsm, k, 13);
#endif
}
