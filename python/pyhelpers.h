#pragma once
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "aesctr/wy.h"
#include "minocore/minocore.h"

using namespace minocore;
using namespace pybind11::literals;
namespace py = pybind11;

template<typename VT, bool SO>
py::object sparse2pysr(const blaze::CompressedVector<VT, SO> &_x) {
    const auto &x = *_x;
    size_t nnz = nonZeros(x);
    py::object ret;
    py::array_t<uint32_t> idx(nnz);
    auto ibi = idx.request();
    uint32_t *iptr = (uint32_t *)ibi.ptr;
    if(sizeof(VT) == 4) {
        py::array_t<float> data(nnz);
        auto dbi = data.request();
        float *ptr = (float *)dbi.ptr;
        for(const auto &pair: x) {
            *ptr++ = pair.value();
            *iptr++ = pair.index();
        }
        ret = py::make_tuple(data, idx);
    } else {
        py::array_t<double> data(nnz);
        auto dbi = data.request();
        double *ptr = (double *)dbi.ptr;
        for(const auto &pair: x) {
            *ptr++ = pair.value();
            *iptr++ = pair.index();
        }
        ret = py::make_tuple(data, idx);
    }
    return ret;
}

template<typename SMT, typename SAL>
py::object centers2pylist(const std::vector<SMT, SAL> &ctrs) {
    py::list ret;
    for(const auto &ctr: ctrs)
        ret.append(sparse2pysr(ctr));
    return ret;
}

template<typename T, typename DestT=float>
auto vec2fnp(const T &x) {
    py::array_t<DestT> ret(x.size());
    auto rbi = ret.request();
    std::copy(std::begin(x), std::end(x), (DestT *)rbi.ptr);
    return ret;
}


