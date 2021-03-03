#pragma once
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "aesctr/wy.h"
#include "minicore/minicore.h"

using namespace minicore;
using namespace pybind11::literals;
namespace py = pybind11;

static std::string standardize_dtype(std::string x);

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
        case 'L': case 'l':
            for(Py_ssize_t i = 0; i < bi.shape[0]; ++i) v.emplace_back(trans(blz::make_cv((uint64_t *)bi.ptr + i * bi.shape[1], bi.shape[1])));
        break;
        case 'I': case 'i':
            for(Py_ssize_t i = 0; i < bi.shape[0]; ++i) v.emplace_back(trans(blz::make_cv((uint32_t *)bi.ptr + i * bi.shape[1], bi.shape[1])));
        break;
        case 'B': case 'b':
            for(Py_ssize_t i = 0; i < bi.shape[0]; ++i) v.emplace_back(trans(blz::make_cv((uint8_t *)bi.ptr + i * bi.shape[1], bi.shape[1])));
        break;
        case 'H': case 'h': for(Py_ssize_t i = 0; i < bi.shape[0]; ++i) v.emplace_back(trans(blz::make_cv((uint16_t *)bi.ptr + i * bi.shape[1], bi.shape[1])));
        break;
        case 'f': for(Py_ssize_t i = 0; i < bi.shape[0]; ++i) {v.emplace_back(trans(blz::make_cv((float *)bi.ptr + i * bi.shape[1], bi.shape[1])));}
        break;
        case 'd': for(Py_ssize_t i = 0; i < bi.shape[0]; ++i) {v.emplace_back(trans(blz::make_cv((double *)bi.ptr + i * bi.shape[1], bi.shape[1])));}
        break;
        default: throw std::invalid_argument(std::string("Invalid format string: ") + bi.format);
    }
}

template<typename SMT, typename SAL>
py::object centers2pylist(const std::vector<SMT, SAL> &ctrs) {
    // We're converting the centers into a CSR-notation object in Numpy
    // First, we compute nonzeros for each row, then use an exclusive scan
    blz::DV<uint64_t> nz(ctrs.size());
    OMP_PFOR
    for(size_t i = 0; i < ctrs.size(); ++i) nz[i] = nonZeros(ctrs[i]);
    const size_t nnz = blz::sum(nz), nr = ctrs.size(), nc = ctrs.front().size();
    py::array_t<uint32_t> idx(nnz);
    py::array_t<uint64_t> indptr(nr + 1);
    using DataT = std::conditional_t<(sizeof(blz::ElementType_t<SMT>) <= 4),
                  float, double>;
    py::array_t<DataT> data(nnz);
    auto idxi = idx.request(), ipi = indptr.request(), datai = data.request();
    uint64_t *ipip = (uint64_t *)ipi.ptr;
    *ipip++ = 0;
    uint64_t csum = 0;
    for(size_t i = 0; i < nr; ++i) {
        csum += nz[i];
        *ipip++ = csum;
    }
    // Now that that's done, we copy it over in parallel
    OMP_PFOR
    for(size_t i = 0; i < nr; ++i) {
        const auto start = ((uint64_t *)ipi.ptr)[i], end = ((uint64_t *)ipi.ptr)[i + 1];
        const size_t n = end - start;
        DataT *const ptr = (DataT *)datai.ptr + start;
        uint32_t *const iptr = (uint32_t *)idxi.ptr + start;
        size_t ind = 0;
        if constexpr(!blaze::IsDenseVector_v<SMT>) {
            auto cbeg = ctrs[i].begin();
            for(;ind < n;++ind, ++cbeg) {
                ptr[ind] = cbeg->value();
                iptr[ind] = cbeg->index();
            }
        } else {
            const auto &ctr = ctrs[i];
            for(size_t j = 0; j < ctr.size(); ++j)
                if(ctr[j] > 0.) iptr[ind] = j, ptr[ind] = ctr[j], ++ind;
        }
    }
    py::array_t<uint64_t> shape(2);
    auto dat = shape.request();
    uint64_t *dptr = (uint64_t *)dat.ptr;
    dptr[0] = nr; dptr[1] = nc;
    return py::make_tuple(py::make_tuple(data, idx, indptr), // Returned matrix in CSR notation
                          shape);   // Shape: num rows by number of columns
}

template<typename T, typename DestT=float>
auto vec2fnp(const T &x) {
    py::array_t<DestT> ret(x.size());
    std::copy(std::begin(x), std::end(x), (DestT *)ret.request().ptr);
    return ret;
}


template<typename Mat>
inline std::vector<blz::CompressedVector<float, blz::rowVector>> obj2dvec(py::object x, const Mat &mat) {
    std::vector<blz::CompressedVector<float, blz::rowVector>> dvecs;
    if(py::isinstance<py::array>(x)){
        py::array arr = py::cast<py::array>(x);
        auto inf = arr.request();
        if(inf.ndim == 1) {
            return obj2dvec(py::list(x), mat);
        } else if(inf.ndim == 2) {
            set_centers(&dvecs, py::cast<py::array>(x).request());
        }
    } else if(py::isinstance<py::sequence>(x)) {
        auto seq = py::cast<py::sequence>(x);
        if(py::isinstance<py::array>(seq[0])) {
            for(auto item: py::cast<py::sequence>(x)) {
                auto ca = py::cast<py::array>(item);
                auto bi = ca.request();
                auto emp = [&](const auto &x) {dvecs.emplace_back() = trans(x);};
                if(bi.format[0] == 'd') emp(blz::make_cv((double *)bi.ptr, bi.size));
                else if(bi.format[0] == 'f') emp(blz::make_cv((float *)bi.ptr, bi.size));
                else throw std::invalid_argument("Array type must be float or double");
            }
        } else if(py::isinstance<py::int_>(seq[0])) {
            const size_t nc = mat.columns();
            for(auto item: py::cast<py::sequence>(x)) {
                Py_ssize_t rownum = py::cast<py::int_>(item).cast<Py_ssize_t>();
                auto &v = dvecs.emplace_back();
                v.resize(nc);
                auto r = row(mat, rownum);
                v.reserve(nonZeros(r));
                if constexpr(blaze::IsDenseMatrix_v<Mat>) {
                    for(size_t i = 0; i < nc; ++i) {
                        if(r[i] > 0.) v.append(i, r[i]);
                    }
                } else for(const auto &pair: r) v.append(pair.index(), pair.value());
            }
        } else throw std::invalid_argument("Invalid: expected numpy array or list of numpy arrays or list of center ids");
    } else throw std::invalid_argument("centers must be a 2d numpy array or sequence containing numpy arrays of the full vectors or center ids");
    return dvecs;
}
template<typename FT>
std::vector<blz::DynamicVector<FT, blz::rowVector>> obj2dvec(
    py::object x, py::array_t<FT, py::array::c_style | py::array::forcecast> dataset) {
    std::vector<blz::DynamicVector<FT, blz::rowVector>> dvecs;
    if(py::isinstance<py::array>(x) && py::cast<py::array>(x).request().ndim == 2) {
        set_centers(&dvecs, py::cast<py::array>(x).request());
    } else if(py::isinstance<py::sequence>(x)) {
        auto seq = py::cast<py::sequence>(x);
        if(py::isinstance<py::array>(seq[0])) {
            for(auto item: py::cast<py::sequence>(x)) {
                auto ca = py::cast<py::array>(item);
                auto bi = ca.request();
                auto emp = [&](const auto &x) {dvecs.emplace_back() = trans(x);};
                if(bi.format[0] == 'd') emp(blz::make_cv((double *)bi.ptr, bi.size));
                else if(bi.format[0] == 'f') emp(blz::make_cv((float *)bi.ptr, bi.size));
                else throw std::invalid_argument("Array type must be float or double");
            }
        } else if(py::isinstance<py::int_>(seq[0])) {
            auto dbi = dataset.request();
            const size_t nc = dbi.shape[1];
            for(auto item: py::cast<py::sequence>(x)) {
                const py::ssize_t rownum = py::cast<py::int_>(item).cast<Py_ssize_t>();
                auto &v = dvecs.emplace_back();
                v.resize(nc);
                switch(standardize_dtype(dbi.format)[0]) {
                    case 'd': v = trans(blz::make_cv((double *)dbi.ptr + rownum * nc, nc)); break;
                    case 'f': v = trans(blz::make_cv((float *)dbi.ptr + rownum * nc, nc)); break;
                    case 'I': case 'i': v = trans(blz::make_cv((uint32_t *)dbi.ptr + rownum * nc, nc)); break;
                    case 'H': case 'h': v = trans(blz::make_cv((uint16_t *)dbi.ptr + rownum * nc, nc)); break;
                    default: throw std::runtime_error("Center items must be double, float, uint32, or uint16");
                }
            }
        } else throw std::invalid_argument("Invalid: expected numpy array or list of numpy arrays or list of center ids");
    } else throw std::invalid_argument("centers must be a 2d numpy array or sequence containing numpy arrays of the full vectors or center ids");
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
