#include "pyfgc.h"
#include "fgc/matrix_coreset.h"

using CSType = coresets::CoresetSampler<float, uint32_t>;
using FNA =  py::array_t<float, py::array::c_style | py::array::forcecast>;
using DNA =  py::array_t<double, py::array::c_style | py::array::forcecast>;
using INA =  py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;

void init_ex1(py::module &m) {
    py::class_<CSType>(m, "CoresetSampler")
    .def(py::init<>())
    .def("make_sampler", [](
        CSType &cs, size_t ncenters, FNA costs, INA assignments, py::object weights, uint64_t seed)
    {
        py::buffer_info buf1 = costs.request(), asb = assignments.request();
        if(buf1.ndim != 1) throw std::runtime_error("buffer must have one dimension (reshape if necessary)");
        float *wp = nullptr;
        auto p = pybind11::cast<FNA>(weights);
        if(p) {
            auto bufp = p.request();
            wp = (float *)bufp.ptr;
        }
        cs.make_sampler(ncenters, costs.shape(0), (float *)buf1.ptr, (uint32_t *)asb.ptr, wp, seed);
    });
}
