#include "minocore/util/csc.h"
#include "blaze/util/Serialization.h"
#include <getopt.h>

void usage() {
    std::fprintf(stderr, "mtxcat <flags> [files]\n-t: transpose\n-h: usage\n-T: transpose submatrices before concatenation\n-e: remove empty rows/columns\n");
    std::exit(1);
}

int main(int argc, char **argv) {
    bool transpose = false, posttranspose = false, empty = false;
    const char *outfile = "/dev/stdout";
    for(int c;(c = getopt(argc, argv, "o:eTth?")) >= 0;) {
        switch(c) {
            case 'h': usage(); break;
            case 't': transpose = true; break;
            case 'T': posttranspose = true; break;
            case 'e': empty = true; break;
            case 'o': outfile = optarg; break;
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
        std::fprintf(stderr, "%s/%d has %zu/%zu and %zu nonzeros\n", paths[i].data(), i, submats[i].rows(), submats[i].columns(), nonZeros(submats[i]));
        if(posttranspose) blaze::transpose(submats[i]);
    }
    std::fprintf(stderr, "Expected %zu nonzeros\n", nz);
    finalmat.resize(s, submats.front().columns());
    finalmat.reserve(nz);
    size_t finaloff = 0;
    for(auto &submat: submats) {
        for(size_t i = 0; i < submat.rows(); ++i) {
            auto r = row(submat, i);
            auto rb = r.begin(), re = r.end();
            while(rb != re) finalmat.append(i, rb->index(), rb->value()), ++rb;
            finalmat.finalize(i + finaloff);
        }
        finaloff += submat.rows();
        auto tmp(std::move(submat)); // free memory
    }
    std::fprintf(stderr, "concatenated mat has %zu rows, %zu columns and %zu nonzeros\n", finalmat.rows(), finalmat.columns(), blaze::nonZeros(finalmat));
    if(empty) {
        auto [rows, cols] = minocore::util::erase_empty(finalmat);
        if(rows.size() || cols.size()) {
            std::fprintf(stderr, "concatenated and emptied mat has %zu rows, %zu columns and %zu nonzeros\n", finalmat.rows(), finalmat.columns(), blaze::nonZeros(finalmat));
            if(std::strcmp(outfile, "/dev/stdout")) {
                std::FILE *ofp = std::fopen((std::string(outfile) + ".erased").data(), "wb");
                if(!ofp) MN_THROW_RUNTIME("Failed to open file for writing");
                std::fprintf(ofp, "#%zu rows, %zu columns\n", rows.size(), cols.size());
                for(const auto r: rows)
                    std::fprintf(ofp, "%zu,", r);
                std::fputc('\n', ofp);
                for(const auto c: cols)
                    std::fprintf(ofp, "%zu,", c);
                std::fputc('\n', ofp);
                std::fclose(ofp);
            }
        }
    }
    blaze::Archive<std::ofstream> arch(outfile);
    arch << finalmat;
}
