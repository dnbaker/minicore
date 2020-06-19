#include "minocore/util/csc.h"
#include "blaze/util/Serialization.h"
#include <getopt.h>

void usage() {
    std::abort();
}

int main(int argc, char **argv) {
    bool transpose = false;
    for(int c;(c = getopt(argc, argv, "th?")) >= 0;) {
        switch(c) {
            case 'h': usage(); break;
            case 't': transpose = true; break;
        }
    }
    blz::SM<double> finalmat;
    std::vector<std::string> paths(argv + optind, argv + argc);
    std::vector<blz::SM<double>> submats(paths.size());
    size_t s = 0, nz = 0;
    OMP_PFOR
    for(unsigned i = 0; i < paths.size(); ++i) {
        submats[i] = minocore::mtx2sparse<double>(paths[i], transpose);
        OMP_ATOMIC
        s += submats[i].rows();
        OMP_ATOMIC
        nz += nonZeros(submats[i]);
        std::fprintf(stderr, "%s/%d has %zu/%zu and %zu nonzers\n", paths[i].data(), i, submats[i].rows(), submats[i].columns(), nonZeros(submats[i]));
    }
    std::fprintf(stderr, "Expected %zu nonzeros\n", nz);
    finalmat.resize(s, submats.front().columns());
    finalmat.reserve(nz);
    size_t finaloff = 0;
    for(const auto &submat: submats) {
        for(size_t i = 0; i < submat.rows(); ++i) {
            auto r = row(submat, i);
            auto rb = r.begin(), re = r.end();
            while(rb != re) finalmat.append(i, rb->index(), rb->value()), ++rb;
            finalmat.finalize(i + finaloff);
        }
        finaloff += submat.rows();
    }
    std::fprintf(stderr, "final mat has %zu/%zu and %zu nonzeros\n", finalmat.rows(), finalmat.columns(), blaze::nonZeros(finalmat));
    blaze::Archive<std::ofstream> arch("/dev/stdout");
    arch << finalmat;
}
