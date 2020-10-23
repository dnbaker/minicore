#include "pycsparse.h"
#include "smw.h"
}

void init_pycsparse(py::module &m) {

    py::class_<PyCSparseMatrix>("CSparseMatrix").def(py::init<py::object>(), py::arg("sparray"))
    .def("__str__", [](const PyCSparseMatrix &x) {return std::string("CSparseMatrix, ") + std::to_string(x.rows()) + "x" + std::to_string(x.columns()) + ", " + std::to_string(x.nnz());})
    .def("columns", [](const PyCSparseMatrix &x) {return x.columns();})
    .def("rows", [](const PyCSparseMatrix &x) {return x.rows();})
    .def("nnz", [](const PyCSparseMatrix &x) {return x.nnz();})

     m.def("kmeanspp", [](const PyCSparseMatrix &smw, const SumOpts &so, py::object weights) {return py_kmeanspp_so(smw, so, weights);},
    "Computes a selecion of points from the matrix pointed to by smw, returning indexes for selected centers, along with assignments and costs for each point.",
       py::arg("smw"),
       py::arg("opts"),
       py::arg("weights") = py::none()
    );
    m.def("kmeanspp",  [](PyCSparseMatrix &smw, py::object msr, py::int_ k, double gamma_beta, uint64_t seed, unsigned nkmc, unsigned ntimes,
                          Py_ssize_t lspp, bool use_exponential_skips,
                          py::object weights) -> py::object
    {
        return py_kmeanspp_noso(smw, msr, k, gamma_beta, seed, nkmc, ntimes, lspp, use_exponential_skips, weights);
    }, "Computes a selecion of points from the matrix pointed to by smw, returning indexes for selected centers, along with assignments and costs for each point."
       "\nSet nkmc to -1 to perform streaming kmeans++ (kmc2 over the full dataset), which parallelizes better but may yield a lower-quality result.\n",
       py::arg("smw"), py::arg("msr"), py::arg("k"), py::arg("betaprior") = 0., py::arg("seed") = 0, py::arg("nkmc") = 0, py::arg("ntimes") = 1,
       py::arg("lspp") = 0, py::arg("use_exponential_skips") = false,
       py::arg("weights") = py::none()
    );
    m.def("greedy_select",  [](PyCSparseMatrix &smw, const SumOpts &so) {
        std::vector<uint64_t> centers;
        const size_t nr = smw.rows();
        blz::DV<double> rsums;
        const double prior = so.gamma < 0. ? 0.: so.gamma;
        const double prior_sum = prior * smw.columns();
        std::vector<double> dret;
        smw.perform([&](auto &matrix) {
            rsums = sum<blaze::rowwise>(matrix);
        });
        smw.perform([&](auto &matrix) {
            std::tie(centers, dret) = m2greedysel(matrix, so);
        });
        auto ret = py::array(size2dtype(nr), std::vector<Py_ssize_t>{nr});
        py::array_t<double> costs(smw.rows());
        auto rpi = ret.request(), cpi = costs.request();
        std::copy(dret.begin(), dret.end(), cpi);
        switch(size2dtype(nr)[0]) {
            case 'L': std::copy(centers.begin(), centers.end(), (uint64_t *)rpi.ptr); break;
            case 'I': std::copy(centers.begin(), centers.end(), (uint32_t *)rpi.ptr); break;
            case 'H': std::copy(centers.begin(), centers.end(), (uint16_t *)rpi.ptr); break;
            case 'B': std::copy(centers.begin(), centers.end(), (int8_t *)rpi.ptr); break;
        }
        return py::make_tuple(ret, costs);
    }, "Computes a greedy selection of points from the matrix pointed to by smw, returning indexes and a vector of costs for each point. To allow for outliers, use the outlier_fraction parameter of Sumopts.",
       py::arg("smw"), py::arg("sumopts"));
}
