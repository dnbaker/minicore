#include "blaze/Util.h"
#include "minocore/util/csc.h"
#include <getopt.h>

void usage() {
    std::fprintf(stderr, "csc2blz <flags> <input> <outputprefix="">\n-d: use double\n-h: usage\n");
    std::exit(1);
}

int main(int argc, char **argv) {
    bool usedouble = false;
    std::string prefix;
    if(argc > 1) prefix = argv[1];
    for(int c;(c = getopt(argc, argv, "fh?"))>= 0;) {
        switch(c) {
            case 'h': usage(); return 1;
            case 'd': usedouble = true; break;
        }
    }
    blaze::Archive<std::ofstream> arch("out.blaze");
    if(usedouble) {
#define MAIN(type) do {\
    auto mat = minocore::csc2sparse<type>(prefix);\
    std::fprintf(stderr, "parsed matrix\n"); arch << mat;\
    } while(0)
        MAIN(double);
    } else {
        MAIN(float);
    }
}
