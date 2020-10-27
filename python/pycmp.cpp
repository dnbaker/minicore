
#include "pyfgc.h"
#include "smw.h"
#include "pyhelpers.h"
using blaze::unaligned;
using blaze::unpadded;
using blaze::rowwise;
using blaze::unchecked;


void init_cmp(py::module &m) {
    m.def("cmp", [](const SparseMatrixWrapper &lhs, py::array arr, py::object msr, py::object betaprior) {
        auto inf = arr.request();
        const double priorv = betaprior.cast<double>(), priorsum = priorv * lhs.columns();
        if(inf.format.size() != 1) throw std::invalid_argument("Invalid dtype");
        const char dt = inf.format[0];
        const size_t nr = lhs.rows();
        const auto ms = assure_dm(msr);
        blz::DV<float> rsums(lhs.rows());
        blz::DV<double> priorc({priorv});
        lhs.perform([&](const auto &x){rsums = blz::sum<rowwise>(x);});
        if(inf.ndim == 1) {
            if(inf.size != Py_ssize_t(lhs.columns())) throw std::invalid_argument("Array must be of the same dimensionality as the matrix");
            py::array_t<float> ret(nr);
            auto v = blz::make_cv((float *)ret.request().ptr, nr);
            lhs.perform([&](auto &matrix) {
                switch(dt) {
#define CASE_F(char, type) \
                    case char: {\
                        auto ov = blz::make_cv(static_cast<type *>(inf.ptr), nr);\
                        const auto vsum = blz::sum(ov);\
                        v = blz::generate(nr, [vsum,priorsum,ms,&matrix,&rsums,&ov,&priorc](auto x) {return cmp::msr_with_prior(ms, row(matrix, x), ov, priorc, priorsum, rsums[x], vsum);});\
                    } break;
                    CASE_F('f', float)
                    CASE_F('d', double)
                    CASE_F('i', int)
                    CASE_F('I', unsigned)
                    CASE_F('h', int16_t)
                    CASE_F('H', uint16_t)
                    CASE_F('b', int16_t)
                    CASE_F('B', uint16_t)
                    CASE_F('l', int64_t)
                    CASE_F('L', uint64_t)
#undef CASE_F
                    default: throw std::invalid_argument("dtypes supported: d, f, i, I, h, H, b, B, l, L");
                }
            });
            return ret;
        } else if(inf.ndim == 2) {
            const Py_ssize_t nc = inf.shape[1], ndr = inf.shape[0];
            if(nc != Py_ssize_t(lhs.columns()))
                throw std::invalid_argument("Array must be of the same dimensionality as the matrix");
            py::array_t<float> ret(std::vector<Py_ssize_t>{Py_ssize_t(nr), nc});
            blz::CustomMatrix<float, unaligned, unpadded, blz::rowMajor> cm((float *)ret.request().ptr, nr, ndr, inf.strides[0]);
            lhs.perform([&](auto &matrix) {
#define CASE_F(char, type) \
                        case char: {\
                            blaze::CustomMatrix<type, unaligned, unpadded> ocm(static_cast<type *>(inf.ptr), ndr, nc);\
                            const auto cmsums = blz::evaluate(blz::sum<blz::rowwise>(ocm));\
                            cm = blz::generate(nr, ndr, [&](auto x, auto y) -> float {\
                                return cmp::msr_with_prior(ms, blz::row(matrix, x, unchecked), blz::row(ocm, y, unchecked), priorc, priorsum, rsums[x], cmsums[y]);\
                            });\
                        } break;
                    switch(dt) {
                        CASE_F('f', float)
                        CASE_F('d', double)
                        CASE_F('i', int)
                        CASE_F('I', unsigned)
                        CASE_F('h', int16_t)
                        CASE_F('H', uint16_t)
                        CASE_F('b', int16_t)
                        CASE_F('B', uint16_t)
                        CASE_F('l', int64_t)
                        CASE_F('L', uint64_t)
#undef CASE_F
                        default: throw std::invalid_argument("dtypes supported: d, f, i, I, h, H, b, B, l, L");
                    }
                    return 0.;
            });
            return ret;
        } else {
            throw std::invalid_argument("NumPy array expected to have 1 or two dimensions.");
        }
        __builtin_unreachable();
        return py::array_t<float>();
    }, py::arg("matrix"), py::arg("data"), py::arg("msr") = 5, py::arg("betaprior") = 0.);
    m.def("cmp", [](const SparseMatrixWrapper &lhs, const SparseMatrixWrapper &rhs, py::object msr, py::object betaprior) {
        const double priorv = betaprior.cast<double>(), priorsum = priorv * lhs.columns();
        const auto ms = assure_dm(msr);
        blz::DV<float> lrsums(lhs.rows());
        blz::DV<float> rrsums(lhs.rows());
        blz::DV<double> priorc({priorv});
        if(lhs.columns() != rhs.columns()) throw std::invalid_argument("mismatched # columns");
        lhs.perform([&](const auto &x){lrsums = blz::sum<rowwise>(x);});
        rhs.perform([&](const auto &x){rrsums = blz::sum<rowwise>(x);});
        const Py_ssize_t nr = lhs.rows(), nc = rhs.rows();
        py::array ret(py::dtype("f"), std::vector<Py_ssize_t>{nr, nc});
        auto retinf = ret.request();
        blz::CustomMatrix<float, unaligned, unpadded, blz::rowMajor> cm((float *)retinf.ptr, nr, nc, nc);
        if(lhs.is_float()) {
            auto &lhr = lhs.getfloat();
            if(rhs.is_float()) {
                auto &rhr = rhs.getfloat();
                cm = blaze::generate(nr, nc, [&](auto lhid, auto rhid) -> float {
                    return cmp::msr_with_prior(ms, blz::row(lhr, lhid, blz::unchecked), blz::row(rhr, rhid, unchecked), priorc, priorsum, lrsums[lhid], rrsums[rhid]);
                });
            } else {
                auto &rhr = rhs.getdouble();
                cm = blaze::generate(nr, nc, [&](auto lhid, auto rhid) -> float {
                    return cmp::msr_with_prior(ms, blz::row(lhr, lhid, blz::unchecked), blz::row(rhr, rhid, unchecked), priorc, priorsum, lrsums[lhid], rrsums[rhid]);
                });
            }
        } else {
            auto &lhr = lhs.getdouble();
            if(rhs.is_float()) {
                auto &rhr = rhs.getfloat();
                cm = blaze::generate(nr, nc, [&](auto lhid, auto rhid) -> float {
                    return cmp::msr_with_prior(ms, blz::row(lhr, lhid, blz::unchecked), blz::row(rhr, rhid, unchecked), priorc, priorsum, lrsums[lhid], rrsums[rhid]);
                });
            } else {
                auto &rhr = rhs.getdouble();
                cm = blaze::generate(nr, nc, [&](auto lhid, auto rhid) -> float {
                    return cmp::msr_with_prior(ms, blz::row(lhr, lhid, blz::unchecked), blz::row(rhr, rhid, unchecked), priorc, priorsum, lrsums[lhid], rrsums[rhid]);
                });
            }
        }
        return ret;
    }, py::arg("matrix"), py::arg("data"), py::arg("msr") = 5, py::arg("betaprior") = 0.);
}
