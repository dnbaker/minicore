#pragma once
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "aesctr/wy.h"
#include "minicore/minicore.h"

#include "./pyhelpers.h"


void init_smw(py::module &);
void init_pycsparse(py::module &);
void init_cmp(py::module &);
void init_coreset(py::module &);
void init_centroid(py::module &);
void init_hashers(py::module &);
void init_omp_helpers(py::module &m);
void init_clustering(py::module &m);
void init_d2s(py::module &m);
void init_clustering_csr(py::module &m);
void init_clustering_dense(py::module &m);
void init_clustering_soft_csr(py::module &m);
void init_clustering_soft(py::module &m);
void init_pydense(py::module &m);
void init_arrcmp(py::module &m);


// Direct CSR mode (no copying) saves memory, but takes a long time to compile
// define BUILD_CSR_CLUSTERING to 1 or modify this file to enable.
#ifndef BUILD_CSR_CLUSTERING
#define BUILD_CSR_CLUSTERING 1
#endif

using CSType = coresets::CoresetSampler<float, uint32_t>;
using FNA =  py::array_t<float, py::array::c_style | py::array::forcecast>;
using DNA =  py::array_t<double, py::array::c_style | py::array::forcecast>;
using INA =  py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;
using SMF = blz::SM<float>;
using SMD = blz::SM<double>;
namespace dist = minicore::distance;
