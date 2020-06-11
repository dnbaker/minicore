#pragma once
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "aesctr/wy.h"
#include "minocore/minocore.h"
#include "minocore/coreset/matrix_coreset.h"
#include "kspp/ks.h"
using namespace minocore;
namespace py = pybind11;
void init_smw(py::module &);
void init_coreset(py::module &);

using CSType = coresets::CoresetSampler<float, uint32_t>;
using FNA =  py::array_t<float, py::array::c_style | py::array::forcecast>;
using DNA =  py::array_t<double, py::array::c_style | py::array::forcecast>;
using INA =  py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;
using SMF = blz::SM<float>;
using SMD = blz::SM<double>;
