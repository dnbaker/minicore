#include "minocore/dist/applicator.h"
#include "minocore/utility.h"
#include "minocore/wip/gen_kmedian.h"
using namespace minocore;
using namespace blz;

#ifndef FLOAT_TYPE
#define FLOAT_TYPE double
#endif

int main(int argc, char *argv[]) {
    if(std::find_if(argv, argv + argc, [](auto x) {return std::strcmp(x, "-h") == 0 || std::strcmp(x, "--help") == 0;})
       != argv + argc) {
        std::fprintf(stderr, "Usage: %s <max rows[1000]> <mincount[50]>\n", *argv);
        std::exit(1);
    }
    unsigned maxnrows = argc == 1 ? 1000: std::atoi(argv[1]);
    unsigned mincount = argc <= 2 ? 50: std::atoi(argv[2]);
    std::string input;
    if(argc > 2)
        input = argv[3];
    std::ofstream ofs("output.txt");
    auto sparsemat = input.size() ? minocore::mtx2sparse<FLOAT_TYPE>(input)
                                  : minocore::csc2sparse<FLOAT_TYPE>("", true);
    std::vector<unsigned> nonemptyrows;
    size_t i = 0;
    while(nonemptyrows.size() < 25) {
        if(sum(row(sparsemat, i)) >= mincount)
            nonemptyrows.push_back(i);
        ++i;
    }
    blz::SM<FLOAT_TYPE> first25 = rows(sparsemat, nonemptyrows.data(), nonemptyrows.size());
    auto jsd = minocore::jsd::make_jsm_applicator(first25);
    //auto jsddistmat = jsd.make_distance_matrix();
    dm::DistanceMatrix<FLOAT_TYPE> utdm(first25.rows());
    jsd.set_distance_matrix(utdm);
    std::cout << utdm << '\n';
    blz::DynamicMatrix<FLOAT_TYPE> jsd_bnj(first25.rows(), first25.rows(), 0.);
    jsd.set_distance_matrix(jsd_bnj, jsd::JSM, true);
    ofs << jsd_bnj << '\n' << blz::min(jsd_bnj) << '\n' << blaze::max(jsd_bnj) << '\n';
    std::fprintf(stderr, "min/max jsm: %g/%g\n", blz::min(jsd_bnj), blz::max(jsd_bnj));
    jsd.set_distance_matrix(jsd_bnj, jsd::JSD, true);
    ofs << jsd_bnj << '\n' << blz::min(jsd_bnj) << '\n' << blaze::max(jsd_bnj) << '\n';
    std::fprintf(stderr, "min/max jsd: %g/%g\n", blz::min(jsd_bnj), blz::max(jsd_bnj));
    ofs.flush();
    i = 25;
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
    first25 = rows(sparsemat, nonemptyrows.data(), nonemptyrows.size());
    std::fprintf(stderr, "Assigned to sparse matrix\n");

    jsd_bnj.resize(nonemptyrows.size(), nonemptyrows.size(), false);
    jsd_bnj = 0.;
    std::fprintf(stderr, "Assigned return matrix to 0.\n");
    auto jsd2 = minocore::jsd::make_probdiv_applicator(first25);
    auto jsd3 = minocore::jsd::make_probdiv_applicator(first25, minocore::jsd::L1);
    minocore::util::Timer timer("1ksparsejsd");
    jsd2.set_distance_matrix(jsd_bnj, minocore::jsd::JSD);
    timer.report();
    std::fprintf(stderr, "bnj after larger minv: %g. maxv: %g\n", blz::min(jsd_bnj), blz::max(jsd_bnj));
    timer.restart("1ksparseL2");
    jsd2.set_distance_matrix(jsd_bnj, minocore::jsd::L2);
    timer.report();
    std::fprintf(stderr, "bnj after larger minv: %g. maxv: %g\n", blz::min(jsd_bnj), blz::max(jsd_bnj));
    timer.restart("1ksparsewllr");
    jsd2.set_distance_matrix(jsd_bnj, minocore::jsd::WLLR, true);
    timer.report();
    timer.restart("1ksparsekl");
    jsd2.set_distance_matrix(jsd_bnj, minocore::jsd::MKL, true);
    timer.report();
    std::cout << "Multinomial KL\n" << '\n';
    //std::cout << jsd_bnj << '\n';
    timer.restart("1ksparseL1");
    jsd2.set_distance_matrix(jsd_bnj, minocore::jsd::EMD, true);
    timer.report();
    std::cout << "EMD: " << jsd_bnj << '\n';
#if 0
    timer.restart("1ldensejsd");
    blz::DM<FLOAT_TYPE> densefirst25 = first25;
    minocore::make_probdiv_applicator(densefirst25).set_distance_matrix(jsd_bnj);
    timer.report();
#endif
    //ofs << "JS Divergence: \n";
    //ofs << jsd_bnj << '\n';
    ofs.flush();
    std::fprintf(stderr, "Starting jsm\n");
    timer.restart("1ksparsejsm");
    jsd2.set_distance_matrix(jsd_bnj, minocore::jsd::L1);
    timer.report();
    timer.reset();
    ofs << "JS Metric: \n";
    ofs << jsd_bnj << '\n';
    ofs.flush();
    std::fprintf(stderr, "Starting llr\n");
    timer.restart("1ksparsellr");
    jsd2.set_distance_matrix(jsd_bnj, minocore::jsd::LLR);
    timer.report();
    timer.reset();
    ofs << "Hicks-Dyjack LLR \n";
    ofs << jsd_bnj << '\n';
    ofs.flush();
    timer.reset();
    std::fprintf(stderr, "\n\nNumber of cells: %zu\n", nonemptyrows.size());
}
