#include "pyfgc.h"


PYBIND11_MODULE(pyfgc, m) {
    init_smw(m);
    init_coreset(m);
    init_merge(m);
    init_centroid(m);
    init_hashers(m);
    init_omp_helpers(m);
    init_clustering(m);
    m.doc() = "Python bindings for FGC, which allows for calling coreset/clustering code from numpy and converting results back to numpy arrays";
}
