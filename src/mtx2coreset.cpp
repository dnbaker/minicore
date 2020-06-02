#include "minocore/minocore.h"

using namespace minocore;

namespace dist = blz::distance;

void usage() {
    std::fprintf(stderr, "I do stuff\n");
}

template<typename FT>
int m2ccore(std::string in, std::string out, bool load_csr, dist::DissimilarityMeasure dis,
            FT gamma,
            dist::Prior prior, int k, size_t coreset_size)
{
    blz::SM<FT> sm(load_csr ? csc2sparse<FT>(in): mtx2sparse<FT>(in));
    std::string cmd = "mkdir " + out;
    if(int i = std::system(cmd.data())) std::fprintf(stderr, "rc: %d\n", i);
    blz::DV<FT, blaze::rowVector> pc{prior == dist::DIRICHLET ? FT(1): gamma};
    auto app(prior == dist::NONE
        ? jsd::make_probdiv_applicator(sm, dis, prior)
        : jsd::make_probdiv_applicator(sm, dis, prior, &pc)
    );
    return 0;
}

int main(int argc, char **argv) {
    std::string inpath, outpath;
    bool load_csr = false, use_double = true;
    dist::DissimilarityMeasure dis = dist::JSD;
    dist::Prior prior = dist::DIRICHLET;
    double gamma = 1.;
    unsigned k = 10;
    size_t coreset_size = 1000;
    for(int c;(c = getopt(argc, argv, "k:g:MT12NCfh?")) >= 0;) {
        switch(c) {
            case 'h': case '?': usage(); std::exit(1);
            case 'f': use_double = false; break;
            case 'c': coreset_size = std::strtoull(optarg, nullptr, 10); break;
            case 'C': load_csr = true; break;
            case 'g': gamma = std::atof(optarg); prior = dist::GAMMA_BETA; break;
            case 'k': k = std::atoi(optarg); break;
            case '1': dis = dist::L1; break;
            case '2': dis = dist::L2; break;
            case 'T': dis = dist::TVD; break;
            case 'M': dis = dist::MKL; break;
            case 'N': prior = dist::NONE;
        }
    }
    if(outpath.empty()) {
        outpath = "mtx2coreset_output";
    }
    return use_double ? m2ccore<double>(inpath, outpath, load_csr, dis, gamma, prior, k, coreset_size)
                      : m2ccore<float>(inpath, outpath, load_csr, dis, gamma, prior, k, coreset_size);
}
