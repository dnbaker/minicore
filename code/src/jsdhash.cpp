#include "fgc/hash.h"
#include <iostream>
using namespace fgc;

int main() {
    unsigned dim = 20, nhashes = 4, nsamp = 10;
    std::mt19937_64 mt(dim  * nhashes);
    std::normal_distribution<float> gen;
    blz::DM<float> dm = blz::generate(nsamp, dim, [&](auto, auto) {return std::abs(gen(mt));});
    assert(dm.rows() == nsamp);
    assert(dm.columns() == dim);
    std::cout << dm << '\n';
    for(auto r: rowiterator(dm)) r /= blz::sum(r);
    for(double r = 0.025; r <= 0.025; r *= 10) {
        std::cout << "R: " << r << '\n';
        JSDLSHasher<float> jsdhasher(dim, nhashes, r);
        S2JSDLSHasher<float> s2hasher(dim, nhashes, r);
        LSHTable<S2JSDLSHasher<float>> s2table(dim, nhashes, r);
        std::fprintf(stderr, "made entries\n");
        auto hashedmat = s2table.hash(dm);
        std::fprintf(stderr, "hashed s2\n");
        //auto jsdhashedmat = jsdhasher.hash(dm);
        std::cout << "Hashed matrix with 5 columns and (what should have) " << nhashes << "rows: \n" << hashedmat << '\n';
        //std::cout << "Hashed matrix with 5 columns and (what should have) " << nhashes << "rows: \n" << jsdhashedmat << '\n';
        s2table.add(dm);
        auto q = s2table.query(dm);
        for(unsigned i = 0; i < q.size(); ++i) {
            for(const auto &pair: q[i]) {
                std::fprintf(stderr, "query item %u matched reference id %u a total of %u times\n", i, pair.first, pair.second);
#ifndef NDEBUG
                if(pair.first == i) assert(pair.second == nhashes);
#endif
            }
        }
#if 0
        blz::DV<float> dv(dim);
        std::mt19937_64 mt(r);
        dv = blz::generate(dim, [](size_t i) {return i;});
        dv /= blz::sum(dv);
        blz::DV<float> hashes = jsdhasher.hash(dv);
        std::cout << hashes << '\n';
        for(unsigned i = 0; i < 4; ++i) {
            dv = blz::generate(dim, [&](size_t){return gen(mt);});
            dv /= blz::sum(dv);
            std::cout << jsdhasher.hash(dv);
        }
        std::cout << '\n';
#endif
    }
}
