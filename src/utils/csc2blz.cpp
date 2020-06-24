#include "blaze/Util.h"
#include "minocore/util/csc.h"
#include <getopt.h>

void usage() {
    std::fprintf(stderr, "csc2blz <flags> <input> <outpath=out.blz>\n-d: use double\n-e: erase empty rows/columns\n-h: usage\n-t: transpose matrix\n");
    std::exit(1);
}

int main(int argc, char **argv) {
    bool usedouble = false, empty = false, tx = false;
    std::string prefix, outpath = "out.blz";
    for(int c;(c = getopt(argc, argv, "tedh?"))>= 0;) {
        switch(c) {
            case 'h': usage(); return 1;
            case 't': tx = true; break;
            case 'd': usedouble = true; break;
            case 'e': empty = true; break;
        }
    }
    if(argc != optind) {
        prefix = argv[optind];
        if(argc != optind + 1) outpath = argv[optind + 1];
    }
    blaze::Archive<std::ofstream> arch(outpath);
    if(usedouble) {
#define MAIN(type) do {\
    auto mat = minocore::csc2sparse<type>(prefix);\
    if(empty) minocore::util::erase_empty(mat);\
    if(transpose) transpose(mat); \
    std::fprintf(stderr, "parsed matrix\n"); arch << mat;\
    } while(0)
        MAIN(double);
    } else {
        MAIN(float);
    }
}
