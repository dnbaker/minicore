#include "fgc/hash.h"
#include <iostream>
using namespace fgc;

int main() {
    unsigned dim = 20, nhashes = 10;
    std::mt19937_64 mt(dim  * nhashes);
    blz::DM<float> dm = blz::generate(dim, 5, [&](auto, auto) {return double(mt()) / mt.max();});
    for(auto r: rowiterator(dm)) r /= blz::sum(r);
    std::uniform_real_distribution<float> gen;
    for(double r = 0.01; r < 100; r *= 10) {
        std::cout << "R: " << r << '\n';
        JSDLSHasher<float> jsdhasher(dim, nhashes, r);
        S2JSDLSHasher<float> s2hasher(dim, nhashes, r);
        auto hashedmat = s2hasher.hash(dm);
        auto jsdhashedmat = jsdhasher.hash(dm);
        std::cout << "Hashed matrix with 5 rows and (what should have) " << nhashes << "columns: \n" << hashedmat << '\n';
        std::cout << "Hashed matrix with 5 columns and (what should have) " << nhashes << "rows: \n" << jsdhashedmat << '\n';
        blz::DV<float> dv(dim);
        std::mt19937_64 mt(r);
        dv = blz::generate(dim, [](size_t i) {return i;});
        dv /= blz::sum(dv);
        blz::DV<float> hashes = jsdhasher.hash(dv);
        std::cout << hashes << '\n';
        for(unsigned i = 0; i < 4; ++i) {
            dv = blz::generate(dim, [&](size_t){return gen(mt);});
            dv /= blz::sum(dv);
            std::cout << trans(jsdhasher.hash(dv)) << '\n';
        }
    }
}
