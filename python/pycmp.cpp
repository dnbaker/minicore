
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
using blaze::row;

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
                using ET = typename std::decay_t<decltype(matrix)>::ElementType;
                using MsrType = std::conditional_t<std::is_floating_point_v<ET>, ET, std::conditional_t<(sizeof(ET) <= 4), float, double>>;
                switch(dt) {
#define CASE_F(char, type) \
                    case char: {\
                        blz::SV<float> sv(blz::make_cv((type *)inf.ptr, inf.size));\
                        const auto vsum = blz::sum(sv);\
                        v = blz::generate(nr, [vsum,priorsum,ms,&matrix,&rsums,&sv,&priorc](auto x) {return cmp::msr_with_prior<MsrType>(ms, sv, row(matrix, x), priorc, priorsum, vsum, rsums[x]);});\
                    } break;
                    CASE_F('f', float)
                    CASE_F('d', double)
                    case 'i': CASE_F('I', unsigned)
#undef CASE_F
                    default: throw std::invalid_argument("dtypes supported: d, f, i, I");
                }
            });
            return ret;
        } else if(inf.ndim == 2) {
            const Py_ssize_t nc = inf.shape[1], ndr = inf.shape[0];
            if(nc != Py_ssize_t(lhs.columns()))
                throw std::invalid_argument("Array must be of the same dimensionality as the matrix");
            std::fprintf(stderr, "Processing matrix of shape %zu/%zu\n", nc, ndr);
            py::array_t<float> ret(std::vector<Py_ssize_t>{Py_ssize_t(nr), ndr});
            blz::CustomMatrix<float, unaligned, unpadded, blz::rowMajor> cm((float *)ret.request().ptr, nr, ndr);
            lhs.perform([&](auto &matrix) {
                using ET = typename std::decay_t<decltype(matrix)>::ElementType;
                using MsrType = std::conditional_t<std::is_floating_point_v<ET>, ET, std::conditional_t<(sizeof(ET) <= 4), float, double>>;
#define CASE_F(char, type) \
                        case char: {\
                            blaze::CustomMatrix<type, unaligned, unpadded> ocm(static_cast<type *>(inf.ptr), ndr, nc);\
                            const auto cmsums = blz::evaluate(blz::sum<blz::rowwise>(ocm));\
                            blz::SM<float> sv = ocm;\
                            cm = blz::generate(nr, ndr, [&](auto x, auto y) -> float {\
                                return cmp::msr_with_prior<MsrType>(ms, \
                                        blz::row(sv, y, unchecked), \
                                        blz::row(matrix, x, unchecked), \
                                        priorc, priorsum, cmsums[y], rsums[x]);\
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
    }, py::arg("matrix"), py::arg("data"), py::arg("msr") = 2, py::arg("prior") = 0.);
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
        const SparseMatrixWrapper *lhp = &lhs, *rhp = &rhs;
        if(lhs.is_float() != rhs.is_float() && rhs.is_float()) {
            std::swap(lhp, rhp);
        }
        if(lhs.is_float() && rhs.is_float()) {
            auto &lhr = lhs.getfloat();
            auto &rhr = rhs.getfloat();
#define DO_GEN\
            cm = blaze::generate(nr, nc, [&](auto lhid, auto rhid) -> float {\
                return cmp::msr_with_prior<float>(ms, \
                            blz::row(rhr, rhid, unchecked), \
                            blz::row(lhr, lhid, unchecked), \
                            priorc, priorsum,\
                            rrsums[rhid], lrsums[lhid]);\
            });
            DO_GEN
        } else if(lhs.is_double() && rhs.is_double()) {
            auto &lhr = lhs.getdouble();
            auto &rhr = rhs.getdouble();
            DO_GEN
        } else {
            auto &lhr = lhp->getfloat();
            auto &rhr = rhp->getdouble();
            DO_GEN
        }
#undef DO_GEN
        return ret;
    }, py::arg("matrix"), py::arg("data"), py::arg("msr") = 2, py::arg("prior") = 0.);
    m.def("cmp", [](const PyCSparseMatrix &lhs, py::array arr, py::object msr, py::object betaprior) {
        auto inf = arr.request();
        const double priorv = betaprior.cast<double>(), priorsum = priorv * lhs.columns();
        if(inf.format.size() != 1) throw std::invalid_argument("Invalid dtype");
        const char dt = inf.format[0];
        const size_t nr = lhs.rows();
        const auto ms = assure_dm(msr);
        blz::DV<float> rsums(lhs.rows());
        blz::DV<double> priorc({priorv});
        lhs.perform([&](const auto &x){rsums = sum<rowwise>(x);});
        if(inf.ndim == 1) {
            if(inf.size != Py_ssize_t(lhs.columns())) throw std::invalid_argument("Array must be of the same dimensionality as the matrix");
            py::array_t<float> ret(nr);
            auto v = blz::make_cv((float *)ret.request().ptr, nr);
            lhs.perform([&](auto &matrix) {
                using ET = typename std::decay_t<decltype(matrix)>::ElementType;
                using MsrType = std::conditional_t<std::is_floating_point_v<ET>, ET, std::conditional_t<(sizeof(ET) <= 4), float, double>>;
                switch(dt) {
#define CASE_F(char, type) \
                    case char: {\
                        blz::SV<float> sv(blz::make_cv((type *)inf.ptr, inf.size));\
                        const auto vsum = blz::sum(sv);\
                        v = blz::generate(nr, [vsum,priorsum,ms,&matrix,&rsums,&sv,&priorc](auto x) {return cmp::msr_with_prior<MsrType>(ms, sv, row(matrix, x), priorc, priorsum, vsum, rsums[x]);});\
                    } break;
                    CASE_F('f', float)
                    CASE_F('d', double)
                    case 'i': CASE_F('I', unsigned)
#undef CASE_F
                    default: throw std::invalid_argument("dtypes supported: d, f, i, I");
                }
            });
            return ret;
        } else if(inf.ndim == 2) {
            const Py_ssize_t nc = inf.shape[1], ndr = inf.shape[0];
            if(nc != Py_ssize_t(lhs.columns()))
                throw std::invalid_argument("Array must be of the same dimensionality as the matrix");
            std::fprintf(stderr, "Processing matrix of shape %zu/%zu\n", nc, ndr);
            py::array_t<float> ret(std::vector<Py_ssize_t>{Py_ssize_t(nr), ndr});
            blz::CustomMatrix<float, unaligned, unpadded, blz::rowMajor> cm((float *)ret.request().ptr, nr, ndr);
            lhs.perform([&](auto &matrix) {
                using ET = typename std::decay_t<decltype(matrix)>::ElementType;
                using MsrType = std::conditional_t<std::is_floating_point_v<ET>, ET, std::conditional_t<(sizeof(ET) <= 4), float, double>>;
#define CASE_F(char, type) \
                        case char: {\
                            blaze::CustomMatrix<type, unaligned, unpadded> ocm(static_cast<type *>(inf.ptr), ndr, nc);\
                            const auto cmsums = blz::evaluate(blz::sum<blz::rowwise>(ocm));\
                            blz::SM<float> sv = ocm;\
                            cm = blz::generate(nr, ndr, [&](auto x, auto y) -> float {\
                                return cmp::msr_with_prior<MsrType>(ms, row(sv, y, unchecked), row(matrix, x, unchecked), priorc, priorsum, cmsums[y], rsums[x]);\
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
    }, py::arg("matrix"), py::arg("data"), py::arg("msr") = 2, py::arg("prior") = 0.);
    m.def("cmp", [](const PyCSparseMatrix &lhs, const PyCSparseMatrix &rhs, py::object msr, py::object betaprior) {
        if(lhs.data_t_ != rhs.data_t_ || lhs.indices_t_ != rhs.indices_t_ || lhs.indptr_t_ != rhs.indptr_t_) {
            std::string lmsg = std::string("lhs ") + lhs.data_t_ + "," + lhs.indices_t_ + "," + lhs.indptr_t_;
            std::string rmsg = std::string("rhs ") + rhs.data_t_ + "," + rhs.indices_t_ + "," + rhs.indptr_t_;
            throw std::invalid_argument(std::string("mismatched types: ") + lmsg + rmsg);
        }
        const double priorv = betaprior.cast<double>(), priorsum = priorv * lhs.columns();
        const auto ms = assure_dm(msr);
        blz::DV<float> lrsums(lhs.rows());
        blz::DV<float> rrsums(lhs.rows());
        blz::DV<double> priorc({priorv});
        if(lhs.columns() != rhs.columns()) throw std::invalid_argument("mismatched # columns");
        lhs.perform([&](const auto &x){lrsums = sum<rowwise>(x);});
        rhs.perform([&](const auto &x){rrsums = sum<rowwise>(x);});
        const Py_ssize_t nr = lhs.rows(), nc = rhs.rows();
        py::array ret(py::dtype("f"), std::vector<Py_ssize_t>{nr, nc});
        auto retinf = ret.request();
        blz::CustomMatrix<float, unaligned, unpadded, blz::rowMajor> cm((float *)retinf.ptr, nr, nc, nc);
        lhs.perform(rhs, [&](auto &mat, auto &rmat) {
            cm = blz::generate(nr, nc, [&](auto lhid, auto rhid) -> float {
                return cmp::msr_with_prior<float>(ms, row(rmat, rhid), row(mat, lhid), priorc, priorsum, rrsums[rhid], lrsums[lhid]);
            });
        });
        return ret;
    }, py::arg("matrix"), py::arg("data"), py::arg("msr") = 2, py::arg("prior") = 0.);
}
