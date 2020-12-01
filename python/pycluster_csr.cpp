#include "pycluster.h"

#if BUILD_CSR_CLUSTERING

#if 0
py::object cluster1_csr(const PyCSparseMatrix &smw, py::int_ k, double beta,
                 py::object msr, py::object weights, double eps,
                 int ntimes, uint64_t seed, int lspprounds, int kmcrounds, uint64_t kmeansmaxiter)
{
    return func1(smw, k, beta, msr, weights, eps, ntimes, seed, lspprounds, kmcrounds, kmeansmaxiter);
}
#endif

#endif

void init_clustering_csr(py::module &m) {
#if BUILD_CSR_CLUSTERING
    m.def("cluster", [](const PyCSparseMatrix &smw, py::object centers, double beta,
                    py::object msr, py::object weights, double eps,
                    uint64_t kmeansmaxiter,
                        //size_t kmcrounds, int ntimes, int lspprounds,
                    uint64_t seed,
                    Py_ssize_t mbsize, Py_ssize_t ncheckins,
                    Py_ssize_t reseed_count, bool with_rep) -> py::object
    {
        return __py_cluster_from_centers(smw, centers, beta, msr, weights, eps, kmeansmaxiter,
                //kmcrounds, ntimes, lspprounds, seed,
                mbsize, ncheckins, reseed_count, with_rep);
    },
    py::arg("smw"),
    py::arg("centers"),
    py::arg("betaprior") = 0.,
    py::arg("msr") = 5,
    py::arg("weights") = py::none(),
    py::arg("eps") = 1e-10,
    py::arg("maxiter") = 1000,
    py::arg("seed") = 0,
    py::arg("mbsize") = Py_ssize_t(-1),
    py::arg("ncheckins") = Py_ssize_t(-1),
    py::arg("reseed_count") = Py_ssize_t(5),
    py::arg("with_rep") = false
    );

#endif
} // init_clustering_csr
