#include "pycluster_soft.h"

void init_clustering_soft_csr(py::module &m) {
    m.def("scluster", [](const PyCSparseMatrix &smw, py::object centers,
                    py::object measure, double beta, double temp,
                    uint64_t kmeansmaxiter, py::ssize_t mbsize, py::ssize_t mbn,
                    py::object savepref, py::object weights) -> py::object
    {
        void *wptr = nullptr;
        std::string wfmt = "f";
        if(!weights.is_none()) {
            auto inf = py::cast<py::array>(weights).request();
            wfmt = standardize_dtype(inf.format);
            wptr = inf.ptr;
        }
        std::string pref = static_cast<std::string>(py::cast<py::str>(savepref));
        return py_scluster(smw, centers, assure_dm(measure), beta, temp, kmeansmaxiter, mbsize, mbn, pref, wptr);
    },
    py::arg("smw"),
    py::arg("centers"),
    py::arg("msr") = 2,
    py::arg("prior") = 0.,
    py::arg("temp") = 1.,
    py::arg("maxiter") = 1000,
    py::arg("mbsize") = py::ssize_t(-1),
    py::arg("mbn") = py::ssize_t(-1),
    py::arg("savepref") = "",
    py::arg("weights") = py::none()
    );
} // init_clustering_csr
