#include "minocore/util/csc.h"

template<typename IndPtrT, typename IndicesT, typename VT>
void dothing(std::string path) {
    auto read = minocore::csc2sparse<float, IndPtrT, IndicesT, VT>(path);
    std::fprintf(stderr, "nr: %zu. nc: %zu. nnz: %zu\n", read.rows(), read.columns(), read.nonZeros());
}

enum VT {
    U32,
    U64,
    F32,
    F64
};

VT c2v(std::string key) {
    if(key == "u32") return U32;
    if(key == "u64") return U64;
    if(key == "f32") return F32;
    if(key == "f64") return F64;
    throw 1;
    return F64;
}

using namespace minocore;
int main(int argc, char *argv[]) {
    std::string inpath;
    VT ip = U64;
    VT id = U64;
    VT dt = F32;
    for(int c;(c = getopt(argc, argv, "p:i:d:h")) >= 0;) {
        switch(c) {
            case 'p': ip = c2v(optarg); break;
            case 'i': id = c2v(optarg); break;
            case 'd': dt = c2v(optarg); break;
        }
    }
    // Use as ./csctest -pu32 -iu32 -df32 cao_atlas_
    if(optind != argc) inpath = argv[optind];
    if(dt != U32 && dt != F32) throw std::runtime_error("Not supported: datatype other than f32 or u32");
    if(ip == U64) {
        if(id == U64) {
            if(dt == U32) {
                dothing<uint64_t, uint64_t, uint32_t>(inpath);
            } else if(dt == F32) {
                dothing<uint64_t, uint64_t, float>(inpath);
            }
        } else {
            if(dt == U32) {
                dothing<uint64_t, uint32_t, uint32_t>(inpath);
            } else if(dt == F32) {
                dothing<uint64_t, uint32_t, float>(inpath);
            }
        }
    } else {
        if(id == U64) {
            if(dt == U32) {
                dothing<uint32_t, uint64_t, uint32_t>(inpath);
            } else if(dt == F32) {
                dothing<uint32_t, uint64_t, float>(inpath);
            }
        } else {
            if(dt == U32) {
                dothing<uint32_t, uint32_t, uint32_t>(inpath);
            } else if(dt == F32) {
                dothing<uint32_t, uint32_t, float>(inpath);
            }
        }
    }
}
