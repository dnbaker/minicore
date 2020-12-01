#include "pyfgc.h"


PYBIND11_MODULE(pyminicore, m) {
    init_smw(m);
    init_coreset(m);
    init_merge(m);
    init_centroid(m);
    //init_hashers(m);
    init_omp_helpers(m);
    init_clustering(m);
    init_pycsparse(m);
    init_clustering_csr(m);
    init_clustering_soft_csr(m);
    init_clustering_soft(m);
    init_cmp(m);
    m.doc() = "Python bindings for FGC, which allows for calling coreset/clustering code from numpy and converting results back to numpy arrays";
}
