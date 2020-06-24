#include "minocore/util/csc.h"
#include <iostream>
#include "blaze/util/Serialization.h"
#include <getopt.h>

int main(int argc, char *argv[]) {
    bool use_float = true, empty = false;
    for(int c;(c = getopt(argc, argv, "edh?")) >= 0;) {
        if(c == 'd') use_float = false;
        else if(c == 'e') empty = true;
    }
    std::string out("/dev/stdout");
    if(optind + 2 <= argc)
        out = argv[optind + 1];
    blaze::Archive<std::ofstream> ret(out);
    if(use_float) {
        auto mat = minocore::util::mtx2sparse2<float>(argc == 1 ? "/dev/stdin": (char *)argv[optind]);
        if(empty) minocore::util::erase_empty(mat);
        ret << mat;
    } else {
        auto mat = minocore::util::mtx2sparse2<double>(argc == 1 ? "/dev/stdin": (char *)argv[optind]);
        if(empty) minocore::util::erase_empty(mat);
        ret << mat;
    }
    return 0;
}
