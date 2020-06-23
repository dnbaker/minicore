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
        auto nr = submats[i].rows();
        OMP_ATOMIC
        s += nr;
        auto lnz = nonZeros(submats[i]);
        OMP_ATOMIC
        nz += lnz;
        std::fprintf(stderr, "%zu/%zu\n", i, paths.size());
        std::fprintf(stderr, "%s/%d has %zu/%zu and %zu nonzeros\n", paths[i].data(), i, submats[i].rows(), submats[i].columns(), nonZeros(submats[i]));
    }
    std::fprintf(stderr, "Finished loop\n");
    if(posttranspose) {
        for(auto &mat: submats) {
            std::fprintf(stderr, "transposing matrix\n");
            blaze::transpose(mat);
            std::fprintf(stderr, "transposed matrix\n");
        }
    }
    std::fprintf(stderr, "Checking column numbers\n");
    for(unsigned i = 0; i < submats.size() - 1; ++i) {
        auto v1 = submats[i].columns(), v2 = submats[i + 1].columns();
        if(v1 != v2) {
            char buf[1024];
            std::sprintf(buf, "Mismatched column numbers: %zu/%zu\n", v1, v2);
            throw std::runtime_error(buf);
        }
    }
    std::fprintf(stderr, "Expected %zu nonzeros\n", nz);
    finalmat.resize(s, submats.front().columns());
    finalmat.reserve(nz);
    size_t finaloff = 0;
    size_t submatid = 0;
    std::reverse(submats.begin(), submats.end());
    while(submats.size()) {
        auto submat = std::move(submats.back());
        submats.pop_back();
        assert(submat.columns() == finalmat.columns());
        for(size_t i = 0; i < submat.rows(); ++i) {
            auto r = row(submat, i);
            auto rb = r.begin(), re = r.end();
            std::fprintf(stderr, "[%zu] row %zu/%zu has %zu nonzeros\n", submatid, i, submat.rows(), nonZeros(r));
            while(rb != re) {
                assert(rb->index() < finalmat.columns());
                finalmat.append(i, rb->index(), rb->value()), ++rb;
            }
            finalmat.finalize(i + finaloff);
        }
        finaloff += submat.rows();
        std::fprintf(stderr, "finaloff after mat %zu is %zu\n", submatid + 1, finaloff);
        ++submatid;
    }
    std::fprintf(stderr, "Made final mat. Free unused memory\n");
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
    std::fprintf(stderr, "Making archive at %s\n", outfile);
    blaze::Archive<std::ofstream> arch(outfile);
    arch << finalmat;
}
