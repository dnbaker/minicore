#include "pyfgc.h"


using CSType = coresets::CoresetSampler<float, uint32_t>;
using FNA =  py::array_t<float, py::array::c_style | py::array::forcecast>;
using DNA =  py::array_t<double, py::array::c_style | py::array::forcecast>;
using INA =  py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;
using SMF = blz::SM<float>;
using SMD = blz::SM<double>;

PYBIND11_MODULE(pyfgc, m) {
    init_ex1(m);
    init_coreset(m);
    m.doc() = "Python bindings for FGC, which allows for calling coreset/clustering code from numpy and converting results back to numpy arrays";
}
void init_ex1(py::module &) {
}
void init_coreset(py::module &m) {
    py::class_<CSType>(m, "CoresetSampler")
    .def(py::init<>())
    .def("make_sampler", [](
        CSType &cs, size_t ncenters, py::array costs, INA assignments, py::object weights, uint64_t seed, int sens_)
    {
        const auto sens(static_cast<minocore::coresets::SensitivityMethod>(sens_));
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
    py::arg("weights") = py::cast<py::none>(Py_None), py::arg("seed") = 13, py::arg("sens")=0
    );
}
