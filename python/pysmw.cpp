#include "smw.h"
#include "pyfgc.h"

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
    });
}
