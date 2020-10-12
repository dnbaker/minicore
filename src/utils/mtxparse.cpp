#include "minicore/util/csc.h"
#include <iostream>
#include "blaze/util/Serialization.h"
#include <getopt.h>

void usage() {
    std::fprintf(stderr, "Usage: mtxparse <flags> input [emits to stdout]\n-d: use doubles, not float.\n-e: empty (removing columns/rows that are empty)\n-r: Emit human-readable form\n");
}

template<typename FT>
void write(std::string in, std::string path, bool empty, bool humanreadable) {
    auto mat = minicore::util::mtx2sparse<FT>(in.data());
    if(empty)
        minicore::util::erase_empty(mat);
    if(humanreadable) {
        std::ofstream ofs(path);
        ofs << mat;
    } else {
        blaze::Archive<std::ofstream> rch(path);
        rch << mat;
    }
}

int main(int argc, char *argv[]) {
    bool use_float = true, empty = false, humanreadable=false;
    const char *in = "/dev/stdin";
    for(int c;(c = getopt(argc, argv, "redh?")) >= 0;) {
        if(c == 'd') use_float = false;
        else if(c == 'e') empty = true;
        else if(c == 'r') humanreadable = true;
        else usage();
    }
    std::string out("/dev/stdout");
    if(optind + 1 <= argc)
        in = argv[optind];
    if(optind + 2 <= argc)
        out = argv[optind + 1];
    if(use_float) {
        write<float>(in, out, empty, humanreadable);
    } else {
        write<double>(in, out, empty, humanreadable);
    }
    return 0;
}
