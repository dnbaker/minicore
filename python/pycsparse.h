#ifndef PYCSPARSEMAT_H
#define PYCSPARSEMAT_H
#include "pyfgc.h"
#include "blaze/util/Serialization.h"
#include "minicore/util/csc.h"

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
        dataap_((void *)data),
        indicesp_((void *)indices),
        indptrp_((void *)indptr),
        shapep_((void *)shape),
        data_t_(py::format_descriptor<DataT>::format()),
        indices_t_(py::format_descriptor<IndicesT>::format()),
        indptr_t_(py::format_descriptor<IndPtrTT>::format()),
        shape_t_(py::format_descriptor<ShapeT>::format()),
        nr_(nr), nc_(nc), nnz_(nnz)
    {
    }
    PyCSparseMatrix(): datap_(nullptr), indicesp_(nullptr), indptrp_(nullptr), nr_(0), nd(0), nnz_(0) {}
    PyCSparseMatrix(const PyCSparseMatrix &) = default;
    PyCSparseMatrix(PyCSparseMatrix &&) = default;
    PyCSparseMatrix &operator=(const PyCSparseMatrix &) = default;
    PyCSparseMatrix &operator=(PyCSparseMatrix &&) = default;
    PyCSparseMatrix(py::object obj): PyCSparseMatrix(py::cast<py::array>(obj.attr("data")), py::cast<py::array>(obj.attr("indices")), py::cast<py::array>(obj.attr("indptr")), py::int_(obj.attr("shape")[0]).cast<Py_ssize_t>(),
                                                    py::int_(obj.attr("shape")[1]).cast<Py_ssize_t>(), obj.attr("nnz").cast<Py_ssize_t>()) {}
    PyCSparseMatrix(py::array data, py::array indices, py::array indptr, Py_ssize_t nr, Py_ssize_t nc, Py_ssize_t nnz): nr_(nr), nc_(nc), nnz_(nnz)
    {
        auto datainf = data.request();
        auto indicesinf = indices.request();
        auto indptrinf = indptr.request();
        datap_ = datainf.ptr;
        indicesp_ = indices.ptr;
        inptrp_ = inptr.ptr;
        data_t_ = data.format;
        indices_t_ = indices.format;
        inptr_t_ = inptr.format;
    }
    // Specialize for Data Types
    template<typename Func> void perform(const Func &func) const {
        switch(data_t_.front()) {
            case 'B': perform_arg<uint8_t, Func>(func); break;
            case 'b': perform_arg<int8_t, Func>(func); break;
            case 'H': perform_arg<uint16_t, Func>(func); break;
            case 'h': perform_arg<int16_t, Func>(func); break;
            case 'L': perform_arg<uint64_t, Func>(func); break;
            case 'l': perform_arg<int64_t, Func>(func); break;
            case 'I': perform_arg<unsigned, Func>(func); break;
            case 'i': perform_arg<int, Func>(func); break;
            case 'd': perform_arg<double, Func>(func); break;
            case 'f': perform_arg<float, Func>(func); break;
            default: throw std::invalid_argument(std::string("Unsupported type for data: ") + data_t_);
        }
    }
    template<typename Func> void perform(const Func &func) {
        switch(data_t_.front()) {
            case 'B': perform_arg<uint8_t, Func>(func); break;
            case 'b': perform_arg<int8_t, Func>(func); break;
            case 'H': perform_arg<uint16_t, Func>(func); break;
            case 'h': perform_arg<int16_t, Func>(func); break;
            case 'L': perform_arg<uint64_t, Func>(func); break;
            case 'l': perform_arg<int64_t, Func>(func); break;
            case 'I': perform_arg<unsigned, Func>(func); break;
            case 'i': perform_arg<int, Func>(func); break;
            case 'd': perform_arg<double, Func>(func); break;
            case 'f': perform_arg<float, Func>(func); break;
            default: throw std::invalid_argument(std::string("Unsupported type for data: ") + data_t_);
        }
    }
    template<typename DataT, typename Func> void perform_arg(const Func &func) const {
        switch(data_t_.front()) {
            case 'B': perform_arg2<DataT, uint8_t, Func>(func); break;
            case 'b': perform_arg2<DataT, int8_t, Func>(func); break;
            case 'H': perform_arg2<DataT, uint16_t, Func>(func); break;
            case 'h': perform_arg2<DataT, int16_t, Func>(func); break;
            case 'L': perform_arg2<DataT, uint64_t, Func>(func); break;
            case 'l': perform_arg2<DataT, int64_t, Func>(func); break;
            case 'I': perform_arg2<DataT, unsigned, Func>(func); break;
            case 'i': perform_arg2<DataT, int, Func>(func); break;
            case 'd': perform_arg2<DataT, double, Func>(func); break;
            case 'f': perform_arg2<DataT, float, Func>(func); break;
            default: throw std::invalid_argument(std::string("Unsupported type for data: ") + data_t_);
        }
    }
    template<typename DataT, typename Func> void perform_arg(const Func &func) {
        switch(data_t_.front()) {
            case 'B': perform_arg2<DataT, uint8_t, Func>(func); break;
            case 'b': perform_arg2<DataT, int8_t, Func>(func); break;
            case 'H': perform_arg2<DataT, uint16_t, Func>(func); break;
            case 'h': perform_arg2<DataT, int16_t, Func>(func); break;
            case 'L': perform_arg2<DataT, uint64_t, Func>(func); break;
            case 'l': perform_arg2<DataT, int64_t, Func>(func); break;
            case 'I': perform_arg2<DataT, unsigned, Func>(func); break;
            case 'i': perform_arg2<DataT, int, Func>(func); break;
            case 'd': perform_arg2<DataT, double, Func>(func); break;
            case 'f': perform_arg2<DataT, float, Func>(func); break;
            default: throw std::invalid_argument(std::string("Unsupported type for data: ") + data_t_);
        }
    }
    template<typename DataT, typename IndicesT, typename Func> void perform_arg2(const Func &func) const {
        switch(data_t_.front()) {
            case 'H': perform_arg3<DataT, IndicesT, uint16_t, Func>(func); break;
            case 'h': perform_arg3<DataT, IndicesT, int16_t, Func>(func); break;
            case 'L': perform_arg3<DataT, IndicesT, uint64_t, Func>(func); break;
            case 'l': perform_arg3<DataT, IndicesT, int64_t, Func>(func); break;
            case 'I': perform_arg3<DataT, IndicesT, unsigned, Func>(func); break;
            case 'i': perform_arg3<DataT, IndicesT, int, Func>(func); break;
            default: throw std::invalid_argument(std::string("Unsupported type for data: ") + data_t_);
        }
    }
    template<typename DataT, typename IndicesT, typename Func> void perform_arg2(const Func &func) {
        switch(data_t_.front()) {
            case 'H': perform_arg3<DataT, IndicesT, uint16_t, Func>(func); break;
            case 'h': perform_arg3<DataT, IndicesT, int16_t, Func>(func); break;
            case 'L': perform_arg3<DataT, IndicesT, uint64_t, Func>(func); break;
            case 'l': perform_arg3<DataT, IndicesT, int64_t, Func>(func); break;
            case 'I': perform_arg3<DataT, IndicesT, unsigned, Func>(func); break;
            case 'i': perform_arg3<DataT, IndicesT, int, Func>(func); break;
            default: throw std::invalid_argument(std::string("Unsupported type for data: ") + data_t_);
        }
    }
    template<typename DataT, typename IndicesT, typename IndPtrT, typename Func>
    void perform_arg3(const Func func) const {
        auto smat = util::make_csparse_matrix((DataT *)datap_, (IndicesT *)indicesp_, (IndPtrT *)indptrp_, nr_, nc_, nnz_);
        func(smat);
    }
    template<typename DataT, typename IndicesT, typename IndPtrT, typename Func>
    void perform(const Func func) {
        auto smat = util::make_csparse_matrix((DataT *)datap_, (IndicesT *)indicesp_, (IndPtrT *)indptrp_, nr_, nc_, nnz_);
        func(smat);
    }
    size_t rows() const {return nr_;}
    size_t columns() const {return nc_;}
    size_t nnz() const {return nnz_;}
};

#endif
