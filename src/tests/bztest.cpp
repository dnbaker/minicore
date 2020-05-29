#undef BLAZE_RANDOM_NUMBER_GENERATOR
#define BLAZE_RANDOM_NUMBER_GENERATOR std::mt19937_64
#include "blaze/Math.h"
#include <iostream>

template<typename MatType, bool SO>
void display(const blaze::Matrix<MatType, SO> &mat) {
    std::cout << mat << '\n';
}

int main() {
    std::srand(13);
    blaze::DynamicMatrix<float, blaze::columnMajor> mat(8, 8);
    randomize(mat);
    display(mat);
    blaze::CustomMatrix<float, blaze::aligned, blaze::unpadded, blaze::columnMajor> m2(&mat(0,0), 8, 8);
    randomize(m2);
    display(m2);
}
