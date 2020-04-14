#pragma once
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "aesctr/wy.h"
#include "fgc/matrix_coreset.h"
using namespace fgc;
namespace py = pybind11;
void init_ex1(py::module &);
void init_coreset(py::module &);
