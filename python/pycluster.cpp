#include "pycluster.h"


void init_clustering(py::module &m) {

    m.def("hcluster", [](SparseMatrixWrapper &smw, py::object centers, double beta,
                         py::object msr, py::object weights, double eps,
                         uint64_t kmeansmaxiter, uint64_t seed, Py_ssize_t mbsize, Py_ssize_t ncheckins,
                         Py_ssize_t reseed_count, bool with_rep, bool use_cs) {
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
    py::arg("mbsize") = Py_ssize_t(-1),
    py::arg("ncheckins") = Py_ssize_t(-1),
    py::arg("reseed_count") = Py_ssize_t(5),
    py::arg("with_rep") = false, py::arg("use_cs") = false,
    "Clusters a SparseMatrixWrapper object using settings and the centers provided above; set prior to < 0 for it to be 1 / ncolumns(). Performs seeding, followed by EM or minibatch k-means");
} // init_clustering
