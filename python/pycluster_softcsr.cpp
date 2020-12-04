#include "pycluster_soft.h"

void init_clustering_soft_csr(py::module &m) {
#if BUILD_CSR_CLUSTERING
    m.def("scluster", [](const PyCSparseMatrix &smw, py::object centers,
                    py::object measure, double beta, double temp,
                    uint64_t kmeansmaxiter, Py_ssize_t mbsize, Py_ssize_t mbn,
                    py::object savepref, bool use_float, py::object weights) -> py::object
    {
        void *wptr = nullptr;
        std::string wfmt = "f";
        if(!weights.is_none()) {
            auto inf = py::cast<py::array>(weights).request();
            wfmt = standardize_dtype(inf.format);
            wptr = inf.ptr;
        }
        std::string pref = static_cast<std::string>(py::cast<py::str>(savepref));
        return py_scluster(smw, centers, assure_dm(measure), beta, temp, kmeansmaxiter, mbsize, mbn, pref, use_float, wptr);
    },
    py::arg("smw"),
    py::arg("centers"),
    py::arg("msr") = 2,
    py::arg("prior") = 0.,
    py::arg("temp") = 1.,
    py::arg("maxiter") = 1000,
    py::arg("mbsize") = Py_ssize_t(-1),
    py::arg("mbn") = Py_ssize_t(-1),
    py::arg("savepref") = "",
    py::arg("use_float") = true,
    py::arg("weights") = py::none()
    );

#endif
} // init_clustering_csr
