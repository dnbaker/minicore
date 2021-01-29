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
namespace dist = minicore::distance;

template<typename VT>
double __ac1d(const VT *__restrict__ lhp, const VT *__restrict__ rhp, DissimilarityMeasure ms, size_t dim, blz::DV<double> &prior, bool reverse, int use_double, double lhsum, double rhsum) {
    if(reverse) return __ac1d(rhp, lhp, ms, dim, prior, !reverse, use_double, rhsum, lhsum);
    double ret;
    const auto psum = prior[0] * dim;
    auto lh = blz::make_cv(const_cast<VT *>(lhp), dim);
    auto rh = blz::make_cv(const_cast<VT *>(rhp), dim);
    if(ms != dist::L1 && ms != dist::L2 && ms != dist::SQRL2) {
        if(lhsum < 0.) lhsum = sum(lh);
        if(rhsum < 0.) rhsum = sum(rh);
    }
    if(use_double) ret = cmp::msr_with_prior<double>(ms, lh, rh, prior, psum, lhsum, rhsum);
    else           ret = cmp::msr_with_prior<float> (ms, lh, rh, prior, psum, lhsum, rhsum);
    return ret;
}

template<typename VT>
void __ac1d2d(const VT *__restrict__ lhp, const VT *__restrict__ rhp, void *ret, DissimilarityMeasure ms, size_t nr, size_t nc, double prior, bool reverse, int use_double,
              double lhsum=-1., double *rhsums=nullptr)
{
    std::unique_ptr<double[]> rhs;
    bool needs_sums = !(ms != dist::L1 && ms != dist::L2 && ms != dist::SQRL2);
    if(needs_sums && rhsums == nullptr) {
        rhs.reset(new double[nr]);
        rhsums = rhs.get();
        for(size_t i = 0; i < nr; ++i) rhsums[i] = sum(blz::make_cv((VT *)rhp + nc * i, nc));
    }
    if(lhsum < 0. && needs_sums) lhsum = sum(blz::make_cv((VT *)lhp, nc));
    blz::DV<double> pv({prior});
    for(size_t i = 0; i < nr; ++i) {
        double rhsum = rhsums ? rhsums[i]: -1.;
        auto v = __ac1d(lhp, rhp + nc * i, ms, nc, pv, reverse, use_double, lhsum, rhsum);
        if(use_double) static_cast<double *>(ret)[i] = v;
        else           static_cast<float *>(ret)[i] = v;
    }
}

template<typename VT>
void __ac2d2d(const VT *__restrict__ lhp, const VT *__restrict__ rhp, void *ret, DissimilarityMeasure ms, size_t lnr, size_t rnr, size_t nc, double prior, bool reverse, int use_double=sizeof(VT) > 4)
{
    blz::DV<double> lhsums(lnr), rhsums(rnr);
    for(size_t i = 0; i < lnr; ++i) lhsums[i] = sum(blz::make_cv((VT *)lhp + i * nc, nc));
    for(size_t i = 0; i < rnr; ++i) rhsums[i] = sum(blz::make_cv((VT *)rhp + i * nc, nc));
    for(size_t i = 0; i < lnr; ++i) {
        void *rptr = static_cast<void *>((float *)ret + rnr * i * (use_double ? 2: 1));
        __ac1d2d(lhp + nc * i, rhp, rptr, ms, rnr, nc, prior, reverse, use_double, lhsums[i], rhsums.data());
    }
}

#if 0
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
#endif

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
    blz::DV<double> pv({prior});
    switch(dt) {
#undef DO_TYPE_
#define DO_TYPE_(c, type)\
        case c: {\
            py::array_t<type, py::array::c_style | py::array::forcecast> lhc(lhs), rhc(rhs);\
            auto lbi = lhc.request(), rbi = rhc.request();\
            return __ac1d((type *)lbi.ptr, (type *)rbi.ptr, ms, lbi.size, pv, reverse, use_double, -1., -1.);\
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
    if(lhi.ndim != rhi.ndim) throw std::runtime_error("arrays should be 1d or 2d");
    auto v = ((lhi.ndim - 1) << 1) | (rhi.ndim - 1);
    switch(v) {
        case 0: return arrcmp1d(lhs, rhs, ms, prior, reverse, use_double, rhdt);
        case 3: return arrcmp2d(lhs, rhs, ms, prior, reverse, use_double, rhdt);
        default: throw std::runtime_error("Wrong number of dimensions");
    }
    return py::none(); // This never happens
}

void init_arrcmp(py::module &m) {
    m.def("cmp", arrcmp, py::arg("lhs"), py::arg("rhs"), py::arg("msr") = 2, py::arg("prior") = 0., py::arg("reverse") = true, py::arg("use_double") = -1);
}
