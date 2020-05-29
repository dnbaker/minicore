#include "minocore/wip/streaming.h"

struct exfunc {
    template<typename...A>
    auto operator()(const A &&...) const {return 1.;}
    template<typename...A>
    auto operator()(A &&...) const {return 1.;}
};

int main() {
    auto clusterer = minocore::streaming::make_kservice_clusterer<uint64_t, exfunc>(exfunc{}, 50, 1e9, 2.);
    clusterer.add(uint64_t(3));
}
