#include "minocore/util/csc.h"

using namespace minocore;
int main(int argc, char *argv[]) {
    std::string inpath;
    if(argc > 1) inpath = argv[1];
    auto read = csc2sparse<float>(inpath);
    std::fprintf(stderr, "nr: %zu. nc: %zu. nnz: %zu\n", read.rows(), read.columns(), read.nonZeros());
}
