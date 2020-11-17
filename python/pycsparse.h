#ifndef PYCSPARSEMAT_H
#define PYCSPARSEMAT_H
#include "pyfgc.h"
#include "blaze/util/Serialization.h"
#include "minicore/util/csc.h"

#define ENABLE_CONST_FUNCS 1

#define ENABLE_NONCONST_FUNCS 0

#ifndef ENABLE_8BITINT_DATA
#define ENABLE_8BITINT_DATA 0
#endif
#ifndef ENABLE_16BITINT_DATA
#define ENABLE_16BITINT_DATA 1
#endif
#ifndef ENABLE_64BITINT_DATA
#define ENABLE_64BITINT_DATA 0
#endif
#ifndef ENABLE_8BITINT_INDICES
#define ENABLE_8BITINT_INDICES 0
#endif
#ifndef ENABLE_16BITINT_INDICES
#define ENABLE_16BITINT_INDICES 0
#endif

struct PyCSparseMatrix {
    void *datap_;
    void *indicesp_;
    void *indptrp_;
    std::string data_t_;
    std::string indices_t_;
    std::string indptr_t_;
    size_t nr_, nc_, nnz_;
    template<typename DataT, typename IndicesT, typename IndPtrT>
    PyCSparseMatrix(DataT *data, const IndicesT *indices, const IndPtrT *indptr, Py_ssize_t nr, Py_ssize_t nc, Py_ssize_t nnz):
        datap_((void *)data),
        indicesp_((void *)indices),
        indptrp_((void *)indptr),
        data_t_(py::format_descriptor<DataT>::format()),
        indices_t_(py::format_descriptor<IndicesT>::format()),
        indptr_t_(py::format_descriptor<IndPtrT>::format()),
        nr_(nr), nc_(nc), nnz_(nnz)
    {
    }
    PyCSparseMatrix(): datap_(nullptr), indicesp_(nullptr), indptrp_(nullptr), nr_(0), nc_(0), nnz_(0) {}
    PyCSparseMatrix(const PyCSparseMatrix &) = default;
    PyCSparseMatrix(PyCSparseMatrix &&) = default;
    PyCSparseMatrix &operator=(const PyCSparseMatrix &) = default;
    PyCSparseMatrix &operator=(PyCSparseMatrix &&) = default;
    PyCSparseMatrix(py::object obj): PyCSparseMatrix(py::cast<py::array>(obj.attr("data")), py::cast<py::array>(obj.attr("indices")), py::cast<py::array>(obj.attr("indptr")), py::int_(py::cast<py::sequence>(obj.attr("shape"))[0]).cast<Py_ssize_t>(),
                                                    py::int_(py::cast<py::sequence>(obj.attr("shape"))[1]).cast<Py_ssize_t>(), obj.attr("nnz").cast<Py_ssize_t>()) {}
    PyCSparseMatrix(py::array data, py::array indices, py::array indptr, Py_ssize_t nr, Py_ssize_t nc, Py_ssize_t nnz): nr_(nr), nc_(nc), nnz_(nnz)
    {
        auto datainf = data.request();
        auto indicesinf = indices.request();
        auto indptrinf = indptr.request();
        datap_ = datainf.ptr;
        indicesp_ = indicesinf.ptr;
        indptrp_ = indptrinf.ptr;
        data_t_ = datainf.format;
        indices_t_ = indicesinf.format;
        indptr_t_ = indptrinf.format;
    }
    // Specialize for Data Types
#if ENABLE_CONST_FUNCS
    template<typename Func> void perform(const Func &func) const {
        switch(data_t_.front()) {
#if ENABLE_8BITINT_DATA
            case 'b': case 'B': _perform<uint8_t,  Func>(func); break;
#endif
#if ENABLE_16BITINT_DATA
            case 'h': case 'H': _perform<uint16_t, Func>(func); break;
#endif
#if ENABLE_64BITINT_DATA
            case 'q': case 'l': case 'u': case 'L': _perform<uint64_t, Func>(func); break;
#endif
            //case 'i': case 'I': _perform<unsigned, Func>(func); break;
            case 'f': _perform<float,    Func>(func); break;
            case 'd': _perform<double,   Func>(func); break;
            default: throw std::invalid_argument(std::string("Unsupported type for data: ") + data_t_);
        }
    }
#endif

#if ENABLE_NONCONST_FUNCS
    template<typename Func> void perform(const Func &func) {
        switch(data_t_.front()) {
#if ENABLE_8BITINT_DATA
            case 'b': case 'B': _perform<uint8_t, Func>(func); break;
#endif
#if ENABLE_16BITINT_DATA
            case 'h': case 'H': _perform<uint16_t, Func>(func); break;
#endif
#if ENABLE_64BITINT_DATA
            case 'q': case 'l': case 'u': case 'L': _perform<uint64_t, Func>(func); break;
#endif
            //case 'i': case 'I': _perform<unsigned, Func>(func); break;
            case 'f': _perform<float, Func>(func); break;
            case 'd': _perform<double, Func>(func); break;
            default: throw std::invalid_argument(std::string("Unsupported type for data: ") + data_t_);
        }
    }
#endif

#define PERF3(c, c2, IPtr, Indices) \
                case c: case c2: {auto smat = util::make_csparse_matrix((DataT *)datap_, (Indices *)indicesp_, (IPtr *)indptrp_, nr_, nc_, nnz_); func(smat);} break

#define PERF2(c, c2, Indices) \
        case c: case c2: { \
            switch(indptr_t_[0]) { \
                PERF3('L', 'l', uint64_t, Indices);\
                PERF3('I', 'i', uint32_t, Indices);\
            }\
        } break

#if ENABLE_CONST_FUNCS
    template<typename DataT, typename Func> void _perform(const Func &func) const {
        switch(indices_t_[0]) {
#if ENABLE_8BITINT_INDICES
            PERF2('B', 'b', uint8_t);
#endif
#if ENABLE_16BITINT_INDICES
            PERF2('H', 'h', uint16_t);
#endif
            PERF2('I', 'i', uint32_t);
            //PERF2('L', 'l', uint64_t);
            default: throw std::invalid_argument(std::string("Unsupported type for indices: ") + indices_t_);
        }
    }
#endif

#if ENABLE_NONCONST_FUNCS
    template<typename DataT, typename Func> void _perform(const Func &func) {
        switch(indices_t_.front()) {
#if ENABLE_8BITINT_INDICES
            PERF2('B', 'b', uint8_t);
#endif
#if ENABLE_16BITINT_INDICES
            PERF2('H', 'h', uint16_t);
#endif
            PERF2('I', 'i', uint32_t);
            //PERF2('L', 'l', uint64_t);
            default: throw std::invalid_argument(std::string("Unsupported type for indices: ") + indices_t_);
        }
    }
#endif
    size_t rows() const {return nr_;}
    size_t columns() const {return nc_;}
    size_t nnz() const {return nnz_;}
};
#undef PERF3
#undef PERF2

#endif
