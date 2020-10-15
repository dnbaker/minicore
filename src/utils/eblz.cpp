#include <iostream>
#include <fstream>
#include "minicore/util/blaze_adaptor.h"
#include "minicore/util/csc.h"
#include "blaze/Util.h"

int main(int argc, char **argv) {
    const char *path = "/dev/stdout";
    bool tx = false;
    for(int c;(c = getopt(argc, argv, "t:h?")) >= 0;) {
        switch(c) {
            case 't': tx = true; break;
            case 'h': std::exit(1); break;
            default: std::abort();
        }
    }
    blaze::Archive<std::ifstream> arch(argv[optind]);
    blaze::CompressedMatrix<double> mat;
    arch >> mat;
    std::cerr << mat.rows() << ", " << mat.columns() << '\n';
    minicore::util::erase_empty(mat);
    if(argc - optind > 1)
        path = argv[optind + 1];

    if(tx) transpose(mat);

    {
        blaze::Archive<std::ofstream> arch(path);
        arch << mat;
    }
}

