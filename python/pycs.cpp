#include "pyfgc.h"
#include "minocore/coreset/matrix_coreset.h"
#include "pybind11/numpy.h"

using CSType = coresets::CoresetSampler<float, uint32_t>;
using FNA =  py::array_t<float, py::array::c_style | py::array::forcecast>;
using DNA =  py::array_t<double, py::array::c_style | py::array::forcecast>;
using INA =  py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;

void init_coreset(py::module &m) {
    py::class_<CSType>(m, "CoresetSampler")
    .def(py::init<>())
    .def("make_sampler", [](
        CSType &cs, size_t ncenters, py::array costs, INA assignments, py::object weights, uint64_t seed, minocore::coresets::SensitivityMethod sens)
    {
        py::buffer_info buf1 = costs.request(), asb = assignments.request();
        if(buf1.ndim != 1) throw std::runtime_error("buffer must have one dimension (reshape if necessary)");
        float *wp = nullptr;
        if(auto p(pybind11::cast<FNA>(weights)); p)
            wp = static_cast<float *>(p.request().ptr);
        if(py::isinstance<py::array_t<float>>(costs)) {
            cs.make_sampler(ncenters, costs.shape(0), (float *)buf1.ptr, (uint32_t *)asb.ptr, wp, seed, sens);
        } else {
            cs.make_sampler(ncenters, costs.shape(0), (double *)buf1.ptr, (uint32_t *)asb.ptr, wp, seed, sens);
        }
    },
    "Generates a coreset sampler given a set of costs, assignments, and, optionally, weights. This can be used to generate an index coreset",
    py::arg("ncenters"), py::arg("costs"), py::arg("assignments"),
    py::arg("weights") =  py::cast<py::none>(Py_None), py::arg("seed") = 13, py::arg("sens")=minocore::coresets::BFL);

}
