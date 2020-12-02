#pragma once
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "aesctr/wy.h"
#include "minicore/minicore.h"

using namespace minicore;
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

template<typename VecT>
void set_centers(VecT *vec, const py::buffer_info &bi) {
    auto &v = *vec;
    switch(bi.format.front()) {
        case 'f':
        for(Py_ssize_t i = 0; i < bi.shape[0]; ++i) {
            auto cv = blz::make_cv((float *)bi.ptr + i * bi.shape[1], bi.shape[1]);
            v.emplace_back(trans(cv));
        }
        break;
        case 'L': case 'l': {
        for(Py_ssize_t i = 0; i < bi.shape[0]; ++i) {
            auto cv = blz::make_cv((uint64_t *)bi.ptr + i * bi.shape[1], bi.shape[1]);
            v.emplace_back(trans(cv));
        }
        }
        break;
        case 'I': case 'i': {
        for(Py_ssize_t i = 0; i < bi.shape[0]; ++i) {
            auto cv = blz::make_cv((int *)bi.ptr + i * bi.shape[1], bi.shape[1]);
            v.emplace_back(trans(cv));
        }
        }
        break;
        case 'B': case 'b': {
        for(Py_ssize_t i = 0; i < bi.shape[0]; ++i) {
            auto cv = blz::make_cv((uint8_t *)bi.ptr + i * bi.shape[1], bi.shape[1]);
            v.emplace_back(trans(cv));
        }
        }
        break;
        case 'H': case 'h': {
        for(Py_ssize_t i = 0; i < bi.shape[0]; ++i) {
            auto cv = blz::make_cv((uint16_t *)bi.ptr + i * bi.shape[1], bi.shape[1]);
            v.emplace_back(trans(cv));
        }
        }
        break;
        case 'd':
        for(Py_ssize_t i = 0; i < bi.shape[0]; ++i) {
            auto cv = blz::make_cv((float *)bi.ptr + i * bi.shape[1], bi.shape[1]);
            v.emplace_back(trans(cv));
        }
        break;
        default: throw std::invalid_argument(std::string("Invalid format string: ") + bi.format);
    }
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


inline std::vector<blz::CompressedVector<float, blz::rowVector>> obj2dvec(py::object x) {
    std::vector<blz::CompressedVector<float, blz::rowVector>> dvecs;
    if(py::isinstance<py::array>(x)) {
        auto cbuf = py::cast<py::array>(x).request();
        set_centers(&dvecs, cbuf);
    } else if(py::isinstance<py::list>(x)) {
        for(auto item: x) {
            auto ca = py::cast<py::array>(item);
            auto bi = ca.request();
            const auto fmt = bi.format[0];
            auto emp = [&](auto &x) {
                dvecs.emplace_back();
                dvecs.back() = trans(x);
            };
            if(fmt == 'd') {
                auto cv = blz::make_cv((double *)bi.ptr, bi.size);
                emp(cv);
            } else {
                auto cv = blz::make_cv((float *)bi.ptr, bi.size);
                emp(cv);
            }
        }
    } else throw std::invalid_argument("centers must be a numpy array or list of numpy arrays");
    return dvecs;
}
inline std::string size2dtype(Py_ssize_t n) {
    if(n > 0xFFFFFFFFu) return "L";
    if(n > 0xFFFFu) return "I";
    if(n > 0xFFu) return "H";
    return "B";
}
static std::string standardize_dtype(std::string x) {
    static const shared::flat_hash_map<std::string, std::string> map {
    {"<f8", "d"},
    {"f8", "d"},
    {"<f4", "f"},
    {"f4", "f"},
    {"<u4", "I"},
    {"u4", "I"},
    {"<i4", "I"},
    {"i4", "I"},
    {"<u8", "L"},
    {"u8", "L"},
    {"<i8", "L"},
    {"i8", "L"},
    {"<u2", "H"},
    {"u2", "H"},
    {"<i2", "H"},
    {"i2", "H"},
    {"<u1", "B"},
    {"u1", "B"},
    {"<i1", "B"},
    {"i1", "B"}
    };
    if(auto it = map.find(x); it != map.end()) x = it->second;
    return x;
}
