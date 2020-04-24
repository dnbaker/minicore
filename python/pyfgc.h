#pragma once
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "aesctr/wy.h"
#include "minocore/minocore.h"
using namespace minocore;
namespace py = pybind11;
void init_ex1(py::module &);
void init_coreset(py::module &);
