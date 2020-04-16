#include "minocore/csc.h"
#include <iostream>
#include "blaze/util/Serialization.h"
#include <getopt.h>

int main(int argc, char *argv[]) {
    bool use_float = true;
    for(int c;(c = getopt(argc, argv, "d:")) >= 0;)
        if(c == 'd') use_float = false;
    std::string out("/dev/stdout");
    if(optind + 2 <= argc)
        out = argv[optind + 1];
    blaze::Archive<std::ofstream> ret(out);
    if(use_float) {
        ret << minocore::mtx2sparse<float>(argc == 1 ? "/dev/stdin": (char *)argv[optind]);
    } else {
        ret << minocore::mtx2sparse<double>(argc == 1 ? "/dev/stdin": (char *)argv[optind]);
    }
    return 0;
}
