#include "pyfgc.h"

void init_arrcmp(py::module &m);

PYBIND11_MODULE(pyminicore, m) {
    init_smw(m);
    init_coreset(m);
    init_centroid(m);
    init_omp_helpers(m);
    init_pycsparse(m);
    init_clustering_csr(m);
    init_clustering_soft_csr(m);
    init_clustering_soft(m);
    init_clustering_dense(m);
    init_clustering(m);
    init_cmp(m);
    init_pydense(m);
    init_arrcmp(m);
    m.doc() = "Python bindings for FGC, which allows for calling coreset/clustering code from numpy and converting results back to numpy arrays";
}
