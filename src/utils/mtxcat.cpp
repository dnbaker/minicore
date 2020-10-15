#include "minicore/util/csc.h"
#include "blaze/util/Serialization.h"
#include <getopt.h>

void usage() {
    std::fprintf(stderr, "mtxcat <flags> [files]\n-t: transpose\n-h: usage\n-T: transpose submatrices before concatenation\n-e: remove empty rows/columns\n");
    std::exit(1);
}

enum EmitFmt {
    BINARY,
    COO
};

int main(int argc, char **argv) {
    minicore::util::TimeStamper ts("argparse");
    bool transpose = false, posttranspose = false, empty = false;
    const char *outfile = "/dev/stdout";
    EmitFmt fmt = BINARY;
    for(int c;(c = getopt(argc, argv, "o:reTth?")) >= 0;) {
        switch(c) {
            case 'h': usage(); break;
            case 't': transpose = true; break;
            case 'T': posttranspose = true; break;
            case 'e': empty = true; break;
            case 'o': outfile = optarg; break;
            case 'r': fmt = COO; break;
        }
    }
    blz::SM<double> finalmat;
    std::vector<std::string> paths(argv + optind, argv + argc);
    std::vector<blz::SM<double>> submats(paths.size());
    size_t s = 0, nz = 0;
    ts.add_event("parse matrices");
    OMP_PFOR
    for(unsigned i = 0; i < paths.size(); ++i) {
        submats[i] = minicore::util::mtx2sparse<double>(paths[i], transpose);
        auto nr = submats[i].rows();
        OMP_ATOMIC
        s += nr;
        auto lnz = nonZeros(submats[i]);
        OMP_ATOMIC
        nz += lnz;
        std::fprintf(stderr, "%u/%zu\n", i, paths.size());
        std::fprintf(stderr, "%s/%d has %zu/%zu and %zu nonzeros\n", paths[i].data(), i, submats[i].rows(), submats[i].columns(), nonZeros(submats[i]));
    }
    std::fprintf(stderr, "Finished loop\n");
    if(posttranspose) {
        ts.add_event("transpose matrices");
        OMP_PFOR
        for(size_t i = 0; i < submats.size(); ++i)
            blaze::transpose(submats[i]);
    }
    ts.add_event("Check matrix sizes");
    std::fprintf(stderr, "Checking column numbers\n");
    for(unsigned i = 0; i < submats.size() - 1; ++i) {
        auto v1 = submats[i].columns(), v2 = submats[i + 1].columns();
        if(v1 != v2) {
            char buf[1024];
            std::sprintf(buf, "Mismatched column numbers: %zu/%zu\n", v1, v2);
            throw std::runtime_error(buf);
        }
    }
    ts.add_event("Resize and reserve");
    std::fprintf(stderr, "Expected %zu nonzeros\n", nz);
    finalmat.resize(s, submats.front().columns());
    finalmat.reserve(nz);
    size_t finaloff = 0;
    std::reverse(submats.begin(), submats.end());
    std::fprintf(stderr, "concatenated mat has %zu rows, %zu columns and %zu nonzeros\n", finalmat.rows(), finalmat.columns(), blaze::nonZeros(finalmat));
    ts.add_event("Construct final");
    while(submats.size()) {
        submatrix(finalmat, finaloff, 0, submats.back().rows(), submats.back().columns()) = submats.back();
        finaloff += submats.back().rows();
        submats.pop_back();
    }
    ts.add_event("Compute row sums");
    auto rsums = blaze::sum<blaze::rowwise>(finalmat);
    ts.add_event("Compute column sums");
    auto csums = blaze::sum<blaze::columnwise>(finalmat);
    std::cerr << "row sums: " << trans(rsums);
    std::cerr << "col sums: " << csums; // to make them emit on one line.
    std::cerr.flush();
    if(empty) {
        ts.add_event("Remove empty features");
        auto [rows, cols] = minicore::util::erase_empty(finalmat);
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
    std::fprintf(stderr, "Making archive at %s\n", outfile);
    ts.add_event("Write to disk");
    if(fmt == BINARY) {
        blaze::Archive<std::ofstream> arch(outfile);
        arch << finalmat;
    } else {
        std::ofstream ofs(outfile);
        ofs << finalmat;
    }
}
