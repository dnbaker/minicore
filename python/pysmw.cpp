#include "smw.h"
#include "pyfgc.h"
#include <sstream>


void init_smw(py::module &m) {
    py::class_<SparseMatrixWrapper>(m, "SparseMatrixWrapper")
    .def(py::init<py::object, py::object, py::object>(), py::arg("sparray"), py::arg("skip_empty")=false, py::arg("use_float")=false)
    .def("is_float", [](SparseMatrixWrapper &wrap) {
        return wrap.is_float();
    })
    .def("is_double", [](SparseMatrixWrapper &wrap) {
        return wrap.is_double();
    }).def("transpose_", [](SparseMatrixWrapper &wrap) {
        wrap.perform([](auto &x){x.transpose();});
    }).def("emit", [](SparseMatrixWrapper &wrap, bool to_stdout) {
        auto func = [to_stdout](auto &x) {
            if(to_stdout) std::cout << x;
            else          std::cerr << x;
        };
        wrap.perform(func);
    }, py::arg("to_stdout")=false)
    .def("__str__", [](SparseMatrixWrapper &wrap) {
        char buf[1024];
        return std::string(buf, std::sprintf(buf, "Matrix of %zu/%zu elements of %s, %zu nonzeros", wrap.rows(), wrap.columns(), wrap.is_float() ? "float32": "double", wrap.nnz()));
    })
    .def("__repr__", [](SparseMatrixWrapper &wrap) {
        wrap.perform([&](auto &x) {
            std::stringstream ss; ss << x;
            return ss.str();
        });
    });


    // Utilities
    m.def("valid_measures", []() {
        py::array_t<uint32_t> ret(sizeof(dist::USABLE_MEASURES) / sizeof(dist::USABLE_MEASURES[0]));
        std::transform(std::begin(dist::USABLE_MEASURES), std::end(dist::USABLE_MEASURES), (uint32_t *)ret.request().ptr, [](auto x) {return static_cast<uint32_t>(x);});
        return ret;
    });
    m.def("meas2desc", [](int x) -> std::string {
        return dist::prob2desc((dist::DissimilarityMeasure)x);
    });
    m.def("meas2str", [](int x) -> std::string {
        return dist::prob2str((dist::DissimilarityMeasure)x);
    });
    m.def("display_measures", [](){
        for(const auto _m: dist::USABLE_MEASURES) {
            std::fprintf(stderr, "%d\t%s\t%s\n", static_cast<int>(_m), prob2str(_m), prob2desc(_m));
        }
    });
    m.def("mdict", []() {
        py::dict ret;
        for(const auto d: dist::USABLE_MEASURES) {
            ret[dist::prob2str(d)] = static_cast<Py_ssize_t>(d);
        }
        return ret;
    });
    py::class_<SumOpts>(m, "sumopts")
    .def(py::init<std::string, Py_ssize_t, double, std::string, double, Py_ssize_t, bool>(), py::arg("measure"), py::arg("k") = 10, py::arg("beta") = 0., py::arg("sm") = "BFL", py::arg("outlier_fraction")=0., py::arg("max_rounds") = 100,
        py::arg("soft") = false)
    .def(py::init<int, Py_ssize_t, double, std::string, double, Py_ssize_t, bool>(), py::arg("measure") = 0, py::arg("k") = 10, py::arg("beta") = 0., py::arg("sm") = "BFL", py::arg("outlier_fraction")=0., py::arg("max_rounds") = 100,
        py::arg("soft") = false)
    .def(py::init<std::string, Py_ssize_t, double, int, double, Py_ssize_t, bool>(), py::arg("measure") = "L1", py::arg("k") = 10, py::arg("beta") = 0., py::arg("sm") = static_cast<int>(minocore::coresets::BFL), py::arg("outlier_fraction")=0., py::arg("max_rounds") = 100,
        py::arg("soft") = false)
    .def(py::init<int, Py_ssize_t, double, int, double, Py_ssize_t, bool>(), py::arg("measure") = 0, py::arg("k") = 10, py::arg("beta") = 0., py::arg("sm") = static_cast<int>(minocore::coresets::BFL), py::arg("outlier_fraction")=0., py::arg("max_rounds") = 100,
        py::arg("soft") = false);
}
