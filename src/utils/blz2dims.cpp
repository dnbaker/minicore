#include "blaze/Math.h"
#include "blaze/Util.h"
#include <iostream>
#include <fstream>

int main(int, char **argv) {
    blaze::Archive<std::ifstream> arch(argv[1]);
    blaze::CompressedMatrix<double> mat;
    arch >> mat;
    std::cerr << mat.rows() << ", " << mat.columns() << '\n';
}
