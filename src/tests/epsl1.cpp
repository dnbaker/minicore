#include "include/minicore/util/proj.h"

int main() {
    blaze::DynamicVector<double> v = blaze::generate(10, [](auto) {return double(std::rand()) / RAND_MAX;});
    minicore::eps_l1(v, 1, 1e-10);
}
