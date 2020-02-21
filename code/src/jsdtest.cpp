#include "fgc/jsd.h"
#include "fgc/csc.h"
#include "fgc/timer.h"
using namespace fgc;

int main(int argc, char *argv[]) {
    if(std::find_if(argv, argv + argc, [](auto x) {return std::strcmp(x, "-h") == 0 || std::strcmp(x, "--help") == 0;})
       != argv + argc) {
        std::fprintf(stderr, "Usage: %s <max rows[1000]> <mincount[50]>\n", *argv);
        std::exit(1);
    }
    unsigned maxnrows = argc == 1 ? 1000: std::atoi(argv[1]);
    unsigned mincount = argc <= 2 ? 50: std::atoi(argv[2]);
    std::ofstream ofs("output.txt");
    auto sparsemat = fgc::csc2sparse("", true);
    std::vector<unsigned> nonemptyrows;
    size_t i = 0;
    while(nonemptyrows.size() < 25) {
        if(sum(row(sparsemat, i)) >= mincount)
            nonemptyrows.push_back(i);
        ++i;
    }
    blz::SM<float> first25 = rows(sparsemat, nonemptyrows.data(), nonemptyrows.size());
#if 0
    auto rws = blz::sum<blz::rowwise>(first25);
    std::cout << rws << '\n';
    std::fprintf(stderr, "First 25 columns: %zu.\n", rws.size());
#endif
    auto jsd = fgc::distance::make_jsm_applicator(first25);
    auto jsddistmat = jsd.make_distance_matrix();
    dm::DistanceMatrix<float> utdm(first25.rows());
    jsd.set_distance_matrix(utdm);
    std::cout << utdm << '\n';
    blz::DynamicMatrix<float> jsd_bnj(first25.rows(), first25.rows(), 0.);
    jsd.set_distance_matrix(jsd_bnj, distance::JSM);
    ofs << jsd_bnj << '\n' << blz::min(jsd_bnj) << '\n' << blaze::max(jsd_bnj) << '\n';
    jsd.set_distance_matrix(jsd_bnj, distance::JSD);
    ofs << jsd_bnj << '\n' << blz::min(jsd_bnj) << '\n' << blaze::max(jsd_bnj) << '\n';
    std::fprintf(stderr, "bnj minv: %g. maxv: %g\n", blz::min(jsd_bnj), blz::max(jsd_bnj));
    ofs.flush();
    auto full_jsd = fgc::distance::make_jsm_applicator(sparsemat);
    wy::WyRand<uint32_t> rng(10);
    double max = full_jsd.llr(rng() % full_jsd.size(), rng() % full_jsd.size()), min = max;
    double jmax = full_jsd.jsd(rng() % full_jsd.size(), rng() % full_jsd.size()), jmin = jmax;
    for(size_t niter = 1000000; niter--;) {
        auto lhs = rng() % full_jsd.size(), rhs = rng() % full_jsd.size();
        if(nonZeros(row(sparsemat, lhs)) == 0 || nonZeros(row(sparsemat, rhs)))
            continue;
        auto bnj = full_jsd.jsd(lhs, rhs);
        auto llr = full_jsd.llr(lhs, rhs);
        assert(llr >= 0.);
        assert(bnj >= 0.);
        //std::fprintf(stdout, "%g\t%g\t%g\n", llr, bnj, std::abs(llr - bnj));
        max = std::max(max, llr);
        min = std::min(min, llr);
        jmax = std::max(jmax, bnj);
        jmin = std::min(jmin, bnj);
    }
    std::fprintf(stderr, "llr max: %g. min: %g\n", max, min);
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
    blaze::DynamicMatrix<float> first25_dense(first25);
    jsd_bnj.resize(nonemptyrows.size(), nonemptyrows.size(), false);
    jsd_bnj = 0.;
    auto jsd2 = fgc::distance::make_probdiv_applicator(first25);
    auto jsd3 = fgc::distance::make_probdiv_applicator(first25_dense);
    fgc::util::Timer timer("1ksparsejsd");
    jsd2.set_distance_matrix(jsd_bnj, fgc::distance::JSD);
    timer.report();
    std::fprintf(stderr, "bnj after larger minv: %g. maxv: %g\n", blz::min(jsd_bnj), blz::max(jsd_bnj));
    timer.restart("1kdensejsd");
    jsd3.set_distance_matrix(jsd_bnj, fgc::distance::JSD);
    timer.report();
    std::fprintf(stderr, "bnj after larger minv: %g. maxv: %g\n", blz::min(jsd_bnj), blz::max(jsd_bnj));
    ofs << "JS Divergence: \n";
    ofs << jsd_bnj << '\n';
    ofs.flush();
    std::fprintf(stderr, "Starting jsm\n");
    timer.restart("1ksparsejsm");
    jsd2.set_distance_matrix(jsd_bnj, fgc::distance::JSM);
    timer.report();
    timer.reset();
    ofs << "JS Metric: \n";
    ofs << jsd_bnj << '\n';
    ofs.flush();
    std::fprintf(stderr, "Starting llr\n");
    timer.restart("1ksparsellr");
    jsd2.set_distance_matrix(jsd_bnj, fgc::distance::LLR);
    timer.report();
    timer.reset();
    ofs << "Hicks-Dyjack LLR \n";
    ofs << jsd_bnj << '\n';
    ofs.flush();
    std::fprintf(stderr, "\n\nNumber of cells: %zu\n", nonemptyrows.size());
}
