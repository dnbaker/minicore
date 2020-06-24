#include "blaze/Util.h"
#include "minocore/util/csc.h"
#include <getopt.h>

void usage() {
    std::fprintf(stderr, "csc2blz <flags> <input> <outputprefix="">\n-d: use double\n-e: erase empty rows/columns\n-h: usage\n");
    std::exit(1);
}

int main(int argc, char **argv) {
    bool usedouble = false, empty = false;
    std::string prefix;
    for(int c;(c = getopt(argc, argv, "edh?"))>= 0;) {
        switch(c) {
            case 'h': usage(); return 1;
            case 'd': usedouble = true; break;
            case 'e': empty = true; break;
        }
    }
    if(argc != optind) prefix = argv[optind];
    blaze::Archive<std::ofstream> arch("out.blaze");
    if(usedouble) {
#define MAIN(type) do {\
    auto mat = minocore::csc2sparse<type>(prefix);\
    if(empty) minocore::util::erase_empty(mat);\
    std::fprintf(stderr, "parsed matrix\n"); arch << mat;\
    } while(0)
        MAIN(double);
    } else {
        MAIN(float);
    }
}
