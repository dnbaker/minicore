#include "pycluster.h"

#if BUILD_CSR_CLUSTERING

#endif

void init_clustering_soft_csr(py::module &m) {
#if BUILD_CSR_CLUSTERING
    m.def("scluster", [](const PyCSparseMatrix &smw, py::object centers, double beta,
                    py::object msr, py::object weights, double eps,
                    uint64_t kmeansmaxiter, size_t kmcrounds, int ntimes, int lspprounds, uint64_t seed, Py_ssize_t mbsize, Py_ssize_t ncheckins,
                    Py_ssize_t reseed_count, bool with_rep, double temp, int use_mmap, std::string saveprefix) -> py::object
    {
        return __py_softcluster_from_centers(smw, centers, beta, msr, weights, eps, kmeansmaxiter, kmcrounds, ntimes, lspprounds, seed, mbsize, ncheckins, reseed_count, with_rep);
    },
    py::arg("smw"),
    py::arg("centers"),
    py::arg("betaprior") = 0.,
    py::arg("msr") = 5,
    py::arg("weights") = py::none(),
    py::arg("eps") = 1e-10,
    py::arg("maxiter") = 1000,
    py::arg("kmcrounds") = 10000,
    py::arg("ntimes") = 1,
    py::arg("lspprounds") = 1,
    py::arg("seed") = 0,
    py::arg("mbsize") = Py_ssize_t(-1),
    py::arg("ncheckins") = Py_ssize_t(-1),
    py::arg("reseed_count") = Py_ssize_t(5),
    py::arg("with_rep") = false,
    py::arg("temp") = 1.,
    py::arg("use_mmap") = false,
    py::arg("use_mmap") = false,
    );

#endif
} // init_clustering_csr
