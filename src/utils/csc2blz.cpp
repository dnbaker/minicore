#include "blaze/Util.h"
#include "minocore/util/csc.h"
#include <getopt.h>

void usage() {
    std::fprintf(stderr, "csc2blz <flags> <input> <outpath=out.blz>\n-d: use floating-point type (vs default int)\n-3: use 32-bit types (64-bit by default).\n-e: erase empty rows/columns\n-h: usage\n-t: transpose matrix\n");
    std::exit(1);
}

enum DT {
    FLT,
    DBL,
    U32,
    U64
};

#define xstr(s) str(s)
#define str(s) #s

int main(int argc, char **argv) {
    bool usefloat = false, empty = false, tx = false, ip64 = true, id64 = true, use32 = false;
    std::string prefix, outpath = "out.blz";
    for(int c;(c = getopt(argc, argv, "pi3tedh?"))>= 0;) {
        switch(c) {
            case 'h': usage(); return 1;
            case 'p': ip64 = true; break;
            case 'i': id64 = true; break;
            case 't': tx = true; break;
            case 'd': usefloat = true; break;
            case 'e': empty = true; break;
            case '3': use32 = true; break;
        }
    }
    if(argc != optind) {
        prefix = argv[optind];
        if(argc != optind + 1) outpath = argv[optind + 1];
    }
    blaze::Archive<std::ofstream> arch(outpath);
#define MAIN(type, ip, id) do {\
    std::cerr << xstr(type) << ", " << xstr(ip) << ", " << xstr(id) << '\n';\
    auto mat = minocore::csc2sparse<type, ip, id>(prefix);\
    if(empty) minocore::util::erase_empty(mat);\
    if(tx) transpose(mat); \
    std::fprintf(stderr, "parsed matrix of shape %zu, %zu\n", mat.rows(), mat.columns()); arch << mat;\
    } while(0)

    switch((usefloat << 3) | (use32 << 2) | (ip64 << 1) | id64) {
        case 0:  MAIN(uint64_t, uint32_t, uint32_t); break;
        case 1:  MAIN(uint64_t, uint32_t, uint64_t); break;
        case 2:  MAIN(uint64_t, uint32_t, uint32_t); break;
        case 3:  MAIN(uint64_t, uint64_t, uint64_t); break;
        case 4:  MAIN(uint32_t, uint32_t, uint32_t); break;
        case 5:  MAIN(uint32_t, uint32_t, uint64_t); break;
        case 6:  MAIN(uint32_t, uint64_t, uint32_t); break;
        case 7:  MAIN(uint32_t, uint64_t, uint64_t); break;
        case 8:  MAIN(double,   uint32_t, uint32_t); break;
        case 9:  MAIN(double,   uint32_t, uint64_t); break;
        case 10: MAIN(double,   uint64_t, uint32_t); break;
        case 11: MAIN(double,   uint64_t, uint64_t); break;
        case 12: MAIN(float,    uint32_t, uint32_t); break;
        case 13: MAIN(float,    uint32_t, uint64_t); break;
        case 14: MAIN(float,    uint64_t, uint32_t); break;
        case 15: MAIN(float,    uint64_t, uint64_t); break;
        default: __builtin_unreachable();
    }
}
