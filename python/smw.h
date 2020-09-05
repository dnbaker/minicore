#ifndef SMW_H
#define SMW_H
#include "pyfgc.h"


struct SparseMatrixWrapper {
private:
    template<typename IndPtrT, typename IndicesT, typename Data>
    SparseMatrixWrapper(const IndPtrT *indptr, const IndicesT *indices, const Data *data,
                  size_t nnz, uint32_t nfeat, uint32_t nitems, bool skip_empty=false, bool use_float=false) {
        if(use_float) {
            matrix_ = csc2sparse<float>(CSCMatrixView<IndPtrT, IndicesT, Data>(indptr, indices, data, nnz, nfeat, nitems), skip_empty);
            auto &m(getfloat());
            std::fprintf(stderr, "[%s] Produced float matrix of %zu/%zu with %zu nonzeros\n", __PRETTY_FUNCTION__, m.rows(), m.columns(), blaze::nonZeros(m));
            std::cerr << m;
        } else {
            matrix_ = csc2sparse<double>(CSCMatrixView<IndPtrT, IndicesT, Data>(indptr, indices, data, nnz, nfeat, nitems), skip_empty);
            auto &m(getdouble());
            std::fprintf(stderr, "[%s] Produced double matrix of %zu/%zu with %zu nonzeros\n", __PRETTY_FUNCTION__, m.rows(), m.columns(), blaze::nonZeros(m));
            std::cerr << m;
        }
    }
    template<typename FT>
    SparseMatrixWrapper(blz::SM<FT> &&mat): matrix_(std::move(mat)) {}
public:
    blz::SM<float> &getfloat() { return std::get<SMF>(matrix_);}
    const blz::SM<float> &getfloat() const { return std::get<SMF>(matrix_);}
    blz::SM<double> &getdouble() { return std::get<SMD>(matrix_);}
    const blz::SM<double> &getdouble() const { return std::get<SMD>(matrix_);}
    template<typename FT>
    SparseMatrixWrapper& operator=(blz::SM<FT> &&mat) {
        if(is_float()) {
            matrix_ = std::move(mat);
        } else {
            {
                SMD tmpmat(std::move(std::get<SMD>(matrix_)));
            }
        }
        return *this;
    }
    size_t nnz() const {
        size_t ret;
        perform([&](auto &x) {ret = blz::nonZeros(x);});
        return ret;
    }
    size_t columns() const {
        size_t ret;
        perform([&](auto &x) {ret = x.columns();});
        return ret;
    }
    size_t rows() const {
        size_t ret;
        perform([&](auto &x) {ret = x.rows();});
        return ret;
    }
    SparseMatrixWrapper(py::object spmat, py::object use_float_py, py::object skip_empty_py) {
        py::array indices = spmat.attr("indices"), indptr = spmat.attr("indptr"), data = spmat.attr("data");
        py::tuple shape = py::cast<py::tuple>(spmat.attr("shape"));
        const bool use_float = py::cast<bool>(use_float_py), skip_empty = py::cast<bool>(skip_empty_py);
        size_t xdim = py::cast<size_t>(shape[0]), ydim = py::cast<size_t>(shape[1]);
        size_t nnz = py::cast<size_t>(spmat.attr("nnz"));
        auto indbuf = indices.request(), indpbuf = indptr.request(), databuf = data.request();
        const void *datptr = databuf.ptr, *indptrptr = indpbuf.ptr, *indicesptr = indbuf.ptr;

#define __DISPATCH(T1, T2, T3) do { \
        std::fprintf(stderr, "Dispatching!\n"); \
        if(use_float) \
            matrix_ = csc2sparse<float>(CSCMatrixView<T1, T2, T3>(reinterpret_cast<const T1 *>(indptrptr), reinterpret_cast<const T2 *>(indicesptr), reinterpret_cast<const T3 *>(datptr), nnz, ydim, xdim), skip_empty); \
        else \
            matrix_ = csc2sparse<double>(CSCMatrixView<T1, T2, T3>(reinterpret_cast<const T1 *>(indptrptr), reinterpret_cast<const T2 *>(indicesptr), reinterpret_cast<const T3 *>(datptr), nnz, ydim, xdim), skip_empty); \
        return; \
    } while(0)
#define __DISPATCH_IF(T1, T2, T3) do { \
                if(py::format_descriptor<T3>::format() == databuf.format) { \
                    std::fprintf(stderr, "T1 %s, T2 %s, with T3 = %s\n", sizeof(T1) == 4 ? "uint": "uint64", sizeof(T2) == 4 ? "uint": "uint64", databuf.format.data());\
                    __DISPATCH(T1, T2, T3); \
                } } while(0)

#define __DISPATCH_ALL_IF(T1, T2) do {\
     __DISPATCH_IF(T1, T2, uint32_t);\
     __DISPATCH_IF(T1, T2, uint64_t);\
     __DISPATCH_IF(T1, T2, int32_t);\
     __DISPATCH_IF(T1, T2, int64_t);\
     __DISPATCH_IF(T1, T2, float);\
     __DISPATCH_IF(T1, T2, double);\
    } while(0)
        if(indbuf.itemsize == 4) {
            if(indpbuf.itemsize == 4) {
                __DISPATCH_ALL_IF(uint32_t, uint32_t);
            } else {
                __DISPATCH_ALL_IF(uint64_t, uint32_t);
            }
        } else {
            assert(indbuf.itemsize == 8);
            if(indpbuf.itemsize == 4) {
                __DISPATCH_ALL_IF(uint32_t, uint64_t);
            } else {
                __DISPATCH_ALL_IF(uint64_t, uint64_t);
            }
        }
        throw std::runtime_error("Unexpected type");
#undef __DISPATCH_ALL_IF
#undef __DISPATCH_IF
#undef __DISPATCH
    }

    std::variant<SMF, SMD> matrix_;
    bool is_float() const {
        assert(is_float() != is_double());
        return std::holds_alternative<SMF>(matrix_);
    }
    bool is_double() const {
        return std::holds_alternative<SMD>(matrix_);
    }
    template<typename Func>
    void perform(const Func &func) {
        if(is_float()) func(std::get<SMF>(matrix_));
        else           func(std::get<SMD>(matrix_));
    }
    template<typename Func>
    void perform(const Func &func) const {
        if(is_float()) func(std::get<SMF>(matrix_));
        else           func(std::get<SMD>(matrix_));
    }
    std::pair<void *, bool> get_opaque() {
        return {is_float() ? static_cast<void *>(&std::get<SMF>(matrix_)): static_cast<void *>(&std::get<SMD>(matrix_)),
                is_float()};
    }
};

#endif
