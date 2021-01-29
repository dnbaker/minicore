#include "pyfgc.h"
#include "smw.h"
#include "pycsparse.h"
#include "pyhelpers.h"
using blaze::unaligned;
using blaze::unpadded;
using blaze::rowwise;
using blaze::unchecked;

using minicore::util::sum;
using minicore::util::row;
using minicore::distance::DissimilarityMeasure;
using blaze::row;

template<typename VT>
double __ac1d(const VT *__restrict__ lhp, const VT *__restrict__ rhp, DissimilarityMeasure ms, size_t dim, double prior, bool reverse, int use_double=sizeof(VT) > 4) {
    if(reverse) return __ac1d(rhp, lhp, ms, dim, prior, !reverse, use_double);
    double ret;
    auto lh = blz::make_cv(const_cast<VT *>(lhp), dim);
    auto rh = blz::make_cv(const_cast<VT *>(rhp), dim);
    const auto lhsum = sum(lh), rhsum = sum(rh);
    const auto psum = prior * dim;
    std::vector<double> pv({prior});
    if(use_double) ret = cmp::msr_with_prior<double>(ms, lh, rh, pv, psum, lhsum, rhsum);
    else           ret = cmp::msr_with_prior<float> (ms, lh, rh, pv, psum, lhsum, rhsum);
    return ret;
}

template<typename VT>
void __ac1d2d(const VT *__restrict__ lhp, const VT *__restrict__ rhp, void *ret, DissimilarityMeasure ms, size_t nr, size_t nc, double prior, bool reverse, int use_double) {
    if(reverse) {
        __ac1d2d(rhp, lhp, ret, ms, nr, nc, prior, !reverse, use_double);
        return;
    }
    for(size_t i = 0; i < nr; ++i) {
        auto v = __ac1d(lhp, rhp + nc * i, ms, nc, prior, reverse, use_double);
        if(use_double) static_cast<double *>(ret)[i] = v;
        else           static_cast<float *>(ret)[i] = v;
    }
}

template<typename VT>
void __ac2d2d(const VT *__restrict__ lhp, const VT *__restrict__ rhp, void *ret, DissimilarityMeasure ms, size_t lnr, size_t rnr, size_t nc, double prior, bool reverse, int use_double=sizeof(VT) > 4)
#if 0
        __ac2d2d((float*)lbi.ptr, (float *)rbi.ptr, ret, ms, lbi.shape[0], rbi.shape[0], lbi.shape[1], prior, reverse, use_double);
#endif
{
    if(reverse) {
        __ac2d2d(rhp, lhp, ret, ms, lnr, rnr, nc, prior, !reverse, use_double);
        return;
    }
    for(size_t i = 0; i < lnr; ++i) {
        void *rptr = static_cast<void *>((float *)ret + rnr * i * (use_double ? 2: 1));
        __ac1d2d(lhp + nc * i, rhp, rptr, ms, rnr, nc, prior, reverse, use_double);
    }
}

py::object arrcmp1d2d(py::array lhs, py::array rhs, DissimilarityMeasure ms, double prior, bool reverse, int use_double, char dt) {
    auto lhi = lhs.request(), rhi = rhs.request();
    if(lhi.ndim != 1) throw std::runtime_error("Expect lh 1");
    if(rhi.ndim != 2) throw std::runtime_error("Expect rh 2");
    py::object ret = py::none();
    if(use_double) ret = py::array_t<double>(rhi.shape[0]);
    else           ret = py::array_t<float>(rhi.shape[0]);
    auto reti = py::cast<py::array>(ret).request();
    switch(dt) {
#define DO_TYPE_(c, type)\
        case c: {\
            py::array_t<type, py::array::c_style | py::array::forcecast> lhc(lhs), rhc(rhs);\
            auto lbi = lhc.request(), rbi = rhc.request();\
            __ac1d2d((type *)lbi.ptr, (type *)rbi.ptr, reti.ptr, ms, rbi.shape[1], lbi.shape[0], prior, reverse, use_double);\
            break;\
        }
        DO_TYPE_('f', float)
        DO_TYPE_('d', double)
        default: ;
    }
    return py::float_(-1.);
}

py::object arrcmp2d(py::array lhs, py::array rhs, DissimilarityMeasure ms, double prior, bool reverse, int use_double, char dt) {
    py::object ret = py::none();
    auto lhi = lhs.request(), rhi = rhs.request();
    if(use_double) ret = py::array_t<double>({py::ssize_t(lhi.shape[0]), py::ssize_t(rhi.shape[0])});
    else           ret = py::array_t<float>({py::ssize_t(lhi.shape[0]), py::ssize_t(rhi.shape[0])});
    py::buffer_info reti = py::cast<py::array>(ret).request();
    if(dt == 'f') {
        py::array_t<float, py::array::c_style | py::array::forcecast> lhc(lhs), rhc(rhs);
        auto lbi = lhc.request(), rbi = rhc.request();
        __ac2d2d((float*)lbi.ptr, (float *)rbi.ptr, reti.ptr, ms, lbi.shape[0], rbi.shape[0], lbi.shape[1], prior, reverse, use_double);
    } else {
        py::array_t<double, py::array::c_style | py::array::forcecast> lhc(lhs), rhc(rhs);
        auto lbi = lhc.request(), rbi = rhc.request();
        __ac2d2d((double*)lbi.ptr, (double *)rbi.ptr, reti.ptr, ms, lbi.shape[0], rbi.shape[0], lbi.shape[1], prior, reverse, use_double);
    }
    return ret;
}

py::float_ arrcmp1d(py::array lhs, py::array rhs, DissimilarityMeasure ms, double prior, bool reverse, int use_double, char dt) {
    switch(dt) {
#undef DO_TYPE_
#define DO_TYPE_(c, type)\
        case c: {\
            py::array_t<type, py::array::c_style | py::array::forcecast> lhc(lhs), rhc(rhs);\
            auto lbi = lhc.request(), rbi = rhc.request();\
            return __ac1d((type *)lbi.ptr, (type *)rbi.ptr, ms, lbi.size, prior, reverse, use_double);\
        }
        DO_TYPE_('f', float)
        DO_TYPE_('d', double)
#undef DO_TYPE_
        default: ;
    }
    return -1.;
}

py::object arrcmp(py::array lhs, py::array rhs, py::object msr, double prior, bool reverse, int use_double=-1) {
    auto lhi = lhs.request(), rhi = rhs.request();
    const auto ms = assure_dm(msr);
    auto lhdt = standardize_dtype(lhi.format)[0], rhdt = standardize_dtype(rhi.format)[0];
    if(lhdt != rhdt) throw std::invalid_argument("arrcmp requires objects be of the same dtype");
    if(use_double < 0) use_double = std::max(lhi.itemsize, rhi.itemsize) > 4;
    if(lhi.ndim > 2 || rhi.ndim > 2) throw std::invalid_argument("arrcmp requires arrays of 1 or 2d");
    auto v = ((lhi.ndim - 1) << 1) | (rhi.ndim - 1);
    switch(v) {
        case 1: case 2: {
            return arrcmp1d2d(lhs, rhs, ms, prior, v > 1 ? reverse: !reverse, use_double, lhdt);
        }
        case 0: return arrcmp1d(lhs, rhs, ms, prior, reverse, use_double, rhdt);
        case 3: return arrcmp2d(lhs, rhs, ms, prior, reverse, use_double, rhdt);
        default: throw std::runtime_error("Wrong number of dimensions");
    }
    return py::none(); // This never happens
}

void init_arrcmp(py::module &m) {
    m.def("cmp", arrcmp, py::arg("lhs"), py::arg("rhs"), py::arg("msr") = 2, py::arg("prior") = 0., py::arg("reverse") = true, py::arg("use_double") = -1);
}
