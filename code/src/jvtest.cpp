#include "fgc/jv.h"

int main() {

    fgc::Graph<boost::undirectedS, float> g;
    std::vector<typename fgc::Graph<boost::undirectedS, float>::vertex_descriptor> vxs(g.vertices().begin(), g.vertices().end());
    if(0) {
        fgc::jain_vazirani_kmedian(g, vxs, 15);
    } else std::fprintf(stderr, "This only checks compilation, not correctness, of JV draft\n");
}
