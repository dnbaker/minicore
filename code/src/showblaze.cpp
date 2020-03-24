#include "blaze/util/Serialization.h"
#include "blaze/Math.h"
#include <fstream>
#include <iostream>

int main(int argc, char *argv[]) {
    if(argc > 1) {
        std::ifstream ifs(argv[1]);
        blaze::Archive<std::ifstream> arch(ifs);
        blaze::CompressedMatrix<float> mat;
        arch >> mat;
        std::cout << mat;
    }
}
