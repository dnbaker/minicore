#include "fgc/jsd.h"
#include "fgc/csc.h"

int main() {
    auto sparsemat = fgc::csc2sparse("", /*normalized=*/true);
    std::vector<unsigned> nonemptyrows;
    size_t i = 0;
    while(nonemptyrows.size() < 25) {
        if(sum(row(sparsemat, i)))
            nonemptyrows.push_back(i);
        ++i;
    }
    blz::SM<float> first25 = rows(sparsemat, nonemptyrows.data(), nonemptyrows.size());
    auto jsd = fgc::jsd::make_jsd_applicator(first25);
    auto jsddistmat = jsd.make_distance_matrix();
    std::cout << jsddistmat << '\n';
    dm::DistanceMatrix<float> utdm(first25.rows());
    jsd.set_distance_matrix(utdm);
    std::cout << utdm << '\n';
}
