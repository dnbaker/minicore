#include "pycluster.h"





#if 0
py::object func1(const SparseMatrixWrapper &smw, py::int_ k, double beta,
                 py::object msr, py::object weights, double eps,
                 int ntimes, uint64_t seed, int lspprounds, int kmcrounds, uint64_t kmeansmaxiter)
{
    const dist::DissimilarityMeasure measure = assure_dm(msr);
    std::fprintf(stderr, "Beginning pycluster (v1)\n");
    if(weights.is_none()) {
        if(smw.is_float())
            return pycluster(smw, k, beta, measure, (blz::DV<float> *)nullptr, eps, ntimes, seed, lspprounds, kmcrounds, kmeansmaxiter);
        else
            return pycluster(smw, k, beta, measure, (blz::DV<double> *)nullptr, eps, ntimes, seed, lspprounds, kmcrounds, kmeansmaxiter);
    }
    auto weightinfo = py::cast<py::array>(weights).request();
    switch(weightinfo.format.front()) {
    case 'b': case 'B': {
        auto cv = blz::make_cv((uint8_t *)weightinfo.ptr, smw.rows());
        return pycluster(smw, k, beta, measure, &cv, eps, ntimes, seed, lspprounds, kmcrounds, kmeansmaxiter);
    }
    case 'h': case 'H': {
        auto cv = blz::make_cv((uint16_t *)weightinfo.ptr, smw.rows());
        return pycluster(smw, k, beta, measure, &cv, eps, ntimes, seed, lspprounds, kmcrounds, kmeansmaxiter);
    }
    case 'u': {
        auto cv = blz::make_cv((unsigned *)weightinfo.ptr, smw.rows());
        return pycluster(smw, k, beta, measure, &cv, eps, ntimes, seed, lspprounds, kmcrounds, kmeansmaxiter);
    }
    case 'i': {
        auto cv = blz::make_cv((int *)weightinfo.ptr, smw.rows());
        return pycluster(smw, k, beta, measure, &cv, eps, ntimes, seed, lspprounds, kmcrounds, kmeansmaxiter);
    }
    case 'f': {
        auto cv = blz::make_cv((float *)weightinfo.ptr, smw.rows());
        return pycluster(smw, k, beta, measure, &cv, eps, ntimes, seed, lspprounds, kmcrounds, kmeansmaxiter);
    }
    case 'd': {
        auto cv = blz::make_cv((double *)weightinfo.ptr, smw.rows());
        return pycluster(smw, k, beta, measure, &cv, eps, ntimes, seed, lspprounds, kmcrounds, kmeansmaxiter);
    }
        default: throw std::invalid_argument(std::string("Unspported weight type: ") + weightinfo.format);
    }
    throw std::invalid_argument("Weights were not float, double, or None.");
}
#endif


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
