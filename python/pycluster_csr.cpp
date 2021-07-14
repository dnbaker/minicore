#include "pycluster.h"

void init_clustering_csr(py::module &m) {
#if BUILD_CSR_CLUSTERING
    m.def("hcluster", [](const PyCSparseMatrix &smw, py::object centers, double beta,
                    py::object msr, py::object weights, double eps,
                    uint64_t kmeansmaxiter,
                    uint64_t seed,
                    py::ssize_t mbsize, py::ssize_t ncheckins,
                    py::ssize_t reseed_count, bool with_rep, bool use_cs) -> py::object
    {
        return __py_cluster_from_centers(smw, centers, beta, msr, weights, eps, kmeansmaxiter,
                seed,
                mbsize, ncheckins, reseed_count, with_rep, use_cs);
    },
    py::arg("smw"),
    py::arg("centers"),
    py::arg("prior") = 0.,
    py::arg("msr") = 2,
    py::arg("weights") = py::none(),
    py::arg("eps") = 1e-10,
    py::arg("maxiter") = 100,
    py::arg("seed") = 0,
    py::arg("mbsize") = py::ssize_t(-1),
    py::arg("ncheckins") = py::ssize_t(-1),
    py::arg("reseed_count") = py::ssize_t(5),
    py::arg("with_rep") = false,
    py::arg("cs") = false
    );

#endif
} // init_clustering_csr
