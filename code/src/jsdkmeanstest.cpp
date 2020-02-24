#include "fgc/jsd.h"
#include "fgc/csc.h"
#include "fgc/timer.h"

int main(int argc, char *argv[]) {
    if(std::find_if(argv, argv + argc, [](auto x) {return std::strcmp(x, "-h") == 0 || std::strcmp(x, "--help") == 0;})
       != argv + argc) {
        std::fprintf(stderr, "Usage: %s <max rows[1000]> <mincount[50]> <k[5]> <m[5000]>\n", *argv);
        std::exit(1);
    }
    unsigned maxnrows = argc == 1 ? 1000: std::atoi(argv[1]);
    unsigned mincount = argc <= 2 ? 50: std::atoi(argv[2]);
    unsigned k        = argc <= 3 ? 50: std::atoi(argv[3]);
    unsigned m        = argc <= 4 ? 5000: std::atoi(argv[4]);
    std::ofstream ofs("output.txt");
    auto sparsemat = fgc::csc2sparse("", true);
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
    auto full_jsd = fgc::distance::make_jsm_applicator(sparsemat);
    fgc::util::Timer timer("k-means");
    auto kmppdat = fgc::distance::make_kmeanspp(full_jsd, k);
    timer.report();
    std::fprintf(stderr, "finished kmpp. Now getting cost\n");
    std::fprintf(stderr, "kmpp solution cost: %g\n", blz::sum(std::get<2>(kmppdat)));
    timer.restart("kmc");
    auto kmcdat = fgc::distance::make_kmc2(full_jsd, k, m);
    timer.report();
    std::fprintf(stderr, "finished kmc2\n");
    timer.reset();
    auto kmc2cost = fgc::coresets::get_oracle_costs(full_jsd, full_jsd.size(), kmcdat);
    std::fprintf(stderr, "kmc2 solution cost: %g\n", blz::sum(kmc2cost.second));
    std::fprintf(stderr, "\n\nNumber of cells: %zu\n", nonemptyrows.size());
    //auto coreset_sampler = fgc::distance::make_d2_coreset_sampler(full_jsd, k, 13);
}
