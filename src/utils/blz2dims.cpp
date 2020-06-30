#include "blaze/Math.h"
#include "blaze/Util.h"
#include <iostream>
#include <fstream>

int main(int, char **argv) {
    blaze::Archive<std::ifstream> arch(argv[1]);
#define TRYTYPE(type) do {\
    try {\
    blaze::CompressedMatrix<double> mat;\
    arch >> mat;\
    std::cerr << mat.rows() << ", " << mat.columns() << '\n';\
    return 0;\
    } catch(...) {}\
    } while(0);
    TRYTYPE(double);
    TRYTYPE(float);
    TRYTYPE(uint8_t);
    TRYTYPE(uint16_t);
    TRYTYPE(uint32_t);
    TRYTYPE(uint64_t);
    TRYTYPE(int8_t);
    TRYTYPE(int16_t);
    TRYTYPE(int32_t);
    TRYTYPE(int64_t);
    throw std::runtime_error("Unrecognized type");
}
