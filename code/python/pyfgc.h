#pragma once
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "aesctr/wy.h"
#include "fgc/matrix_coreset.h"
namespace py = pybind11;
void init_ex1(py::module &);
void init_ex2(py::module &);
