#include "blaze/Util.h"
#include "minicore/util/csc.h"
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
    bool usefloat = false, empty = false, tx = false, ip64 = false, id64 = false, use32 = false;
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
    auto mat = minicore::csc2sparse<type, ip, id, type>(prefix);\
    if(empty) minicore::util::erase_empty(mat);\
    if(tx) transpose(mat); \
    std::fprintf(stderr, "parsed matrix of shape %zu, %zu\n", mat.rows(), mat.columns()); arch << mat;\
    } while(0)

    if(usefloat) {
        if(use32) {
            if(ip64) {
                if(id64) MAIN(float, uint64_t, uint64_t); else MAIN(float, uint64_t, uint32_t);
            } else {
                if(id64) MAIN(float, uint32_t, uint64_t); else MAIN(float, uint32_t, uint32_t);
            }
        } else {
            if(ip64) {
                if(id64) MAIN(double, uint64_t, uint64_t); else MAIN(double, uint64_t, uint32_t);
            } else {
                if(id64) MAIN(double, uint32_t, uint64_t); else MAIN(double, uint32_t, uint32_t);
            }
        }
    } else {
        if(use32) {
            if(ip64) {
                if(id64) MAIN(uint32_t, uint64_t, uint64_t); else MAIN(uint32_t, uint64_t, uint32_t);
            } else {
                if(id64) MAIN(uint32_t, uint32_t, uint64_t); else MAIN(uint32_t, uint32_t, uint32_t);
            }
        } else {
            if(ip64) {
                if(id64) MAIN(uint64_t, uint64_t, uint64_t); else MAIN(uint64_t, uint64_t, uint32_t);
            } else {
                if(id64) MAIN(uint64_t, uint32_t, uint64_t); else MAIN(uint64_t, uint32_t, uint32_t);
            }
        }
    }
}
