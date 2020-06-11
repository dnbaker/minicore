#include "pyfgc.h"
#include "kspp/ks.h"


using CSType = coresets::CoresetSampler<float, uint32_t>;
using FNA =  py::array_t<float, py::array::c_style | py::array::forcecast>;
using DNA =  py::array_t<double, py::array::c_style | py::array::forcecast>;
using INA =  py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;
using SMF = blz::SM<float>;
using SMD = blz::SM<double>;

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
    static py::object ensure_csc(py::object spmat) {
        if(py::hasattr(spmat, "asformat")) {
            throw std::runtime_error("spmat must be a SciPy sparse matrix");
        }
        spmat = spmat("asformat", "csc");
        return spmat;
    }
    SparseMatrixWrapper(py::object spmat, py::object use_float_py, py::object skip_empty_py) {
        spmat = ensure_csc(spmat);
        py::array indices = spmat.attr("indices"), indptr = spmat.attr("indptr"), data = spmat.attr("data");
        py::tuple shape = py::cast<py::tuple>(spmat.attr("shape"));
        const bool use_float = py::cast<bool>(use_float_py), skip_empty = py::cast<bool>(skip_empty_py);
        const void *datptr = data.request().ptr, *indptrptr = indptr.request().ptr, *indicesptr = indices.request().ptr;
        size_t xdim = py::cast<size_t>(shape[0]), ydim = py::cast<size_t>(shape[1]);
        size_t nnz = py::cast<size_t>(spmat.attr("nnz"));

#define __DISPATCH(T1, T2, T3) do { \
        if(use_float) \
            matrix_ = csc2sparse<float>(CSCMatrixView<T1, T2, T3>(reinterpret_cast<const T1 *>(indptrptr), reinterpret_cast<const T2 *>(indicesptr), reinterpret_cast<const T3 *>(datptr), nnz, ydim, xdim), skip_empty); \
        else \
            matrix_ = csc2sparse<double>(CSCMatrixView<T1, T2, T3>(reinterpret_cast<const T1 *>(indptrptr), reinterpret_cast<const T2 *>(indicesptr), reinterpret_cast<const T3 *>(datptr), nnz, ydim, xdim), skip_empty); \
        return; \
    } while(0)

#define __DISPATCH_ALL_T2(T1, T2) do { \
            if(py::isinstance<py::array_t<T2>>(indices)) { \
                if(py::isinstance<py::array_t<float>>(data)) { __DISPATCH(T1, T2, float); \
                } else if(py::isinstance<py::array_t<double>>(data)) { __DISPATCH(T1, T2, double); \
                } else if(py::isinstance<py::array_t<uint8_t>>(data)) { __DISPATCH(T1, T2, uint8_t);\
                } else if(py::isinstance<py::array_t<uint16_t>>(data)) { __DISPATCH(T1, T2, uint16_t);\
                } else if(py::isinstance<py::array_t<uint32_t>>(data)) { __DISPATCH(T1, T2, uint32_t);\
                } else if(py::isinstance<py::array_t<uint64_t>>(data)) { __DISPATCH(T1, T2, uint64_t);\
                } else if(py::isinstance<py::array_t<int8_t>>(data)) { __DISPATCH(T1, T2, int8_t);\
                } else if(py::isinstance<py::array_t<int16_t>>(data)) { __DISPATCH(T1, T2, int16_t);\
                } else if(py::isinstance<py::array_t<int32_t>>(data)) { __DISPATCH(T1, T2, int32_t);\
                } else if(py::isinstance<py::array_t<int64_t>>(data)) { __DISPATCH(T1, uint64_t, int64_t);\
                } else {throw std::runtime_error("Unexpected type");}\
            } \
        } while(0)

#define __DISPATCH_ALL_T1(T1) do { \
            if(py::isinstance<py::array_t<T1>>(indptr)) { \
                } else if(py::isinstance<py::array_t<uint8_t>>(indices)) { __DISPATCH_ALL_T2(T1, uint8_t);\
                } else if(py::isinstance<py::array_t<uint16_t>>(indices)) { __DISPATCH_ALL_T2(T1, uint16_t);\
                } else if(py::isinstance<py::array_t<uint32_t>>(indices)) { __DISPATCH_ALL_T2(T1, uint32_t);\
                } else if(py::isinstance<py::array_t<uint64_t>>(indices)) { __DISPATCH_ALL_T2(T1, uint64_t);\
                } else if(py::isinstance<py::array_t<int8_t>>(indices)) { __DISPATCH_ALL_T2(T1, int8_t);\
                } else if(py::isinstance<py::array_t<int16_t>>(indices)) { __DISPATCH_ALL_T2(T1, int16_t);\
                } else if(py::isinstance<py::array_t<int32_t>>(indices)) { __DISPATCH_ALL_T2(T1, int32_t);\
                } else if(py::isinstance<py::array_t<int64_t>>(indices)) { __DISPATCH_ALL_T2(T1, int64_t);\
                } else {throw std::runtime_error("Unexpected type");\
            } \
        } while(0)

        if(py::isinstance<py::array_t<uint32_t>>(indices)) {
            __DISPATCH_ALL_T1(uint32_t);
        } else if(py::isinstance<py::array_t<uint64_t>>(indices)) {
            __DISPATCH_ALL_T1(uint64_t);
        } else {
            throw std::runtime_error("Unexpected type");
        }
#undef __DISPATCH_ALL_T1
#undef __DISPATCH_ALL_T2
#undef __DISPATCH
    }

    std::variant<SMF, SMD> matrix_;
    bool is_float() const {
        return std::holds_alternative<SMF>(matrix_);
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

PYBIND11_MODULE(pyfgc, m) {
    init_ex1(m);
    init_coreset(m);
    m.doc() = "Python bindings for FGC, which allows for calling coreset/clustering code from numpy and converting results back to numpy arrays";
}
void init_ex1(py::module &) {
}
void init_coreset(py::module &m) {
    py::class_<CSType>(m, "CoresetSampler")
    .def(py::init<>())
    .def("make_sampler", [](
        CSType &cs, size_t ncenters, py::array costs, INA assignments, py::object weights, uint64_t seed, int sens_)
    {
        const auto sens(static_cast<minocore::coresets::SensitivityMethod>(sens_));
        py::buffer_info buf1 = costs.request();
        const uint32_t *asnp = (const uint32_t *)assignments.request().ptr;
        if(buf1.ndim != 1) throw std::runtime_error("buffer must have one dimension (reshape if necessary)");
        float *wp = nullptr;
        if(auto p(pybind11::cast<FNA>(weights)); p)
            wp = static_cast<float *>(p.request().ptr);
        if(py::isinstance<py::array_t<float>>(costs)) {
            cs.make_sampler(ncenters, costs.shape(0), (float *)buf1.ptr, asnp, wp, seed, sens);
        } else {
            cs.make_sampler(ncenters, costs.shape(0), (double *)buf1.ptr, asnp, wp, seed, sens);
        }
    },
    "Generates a coreset sampler given a set of costs, assignments, and, optionally, weights. This can be used to generate an index coreset",
    py::arg("ncenters"), py::arg("costs"), py::arg("assignments"),
    py::arg("weights") = py::cast<py::none>(Py_None), py::arg("seed") = 13, py::arg("sens")=0
    ).def("get_probs", [](CSType &cs) {
        py::array_t<float> ret(cs.np_);
        std::copy(cs.probs_.get(), cs.probs_.get() + cs.np_, (float *)ret.request().ptr);
        return ret;
    }, "Create a numpy array of sampling probabilities");
    py::class_<SparseMatrixWrapper>(m, "SparseMatrixWrapper")
    .def(py::init<py::object, py::object, py::object>(), py::arg("sparray"), py::arg("skip_empty")=false, py::arg("use_float")=false)
    .def("is_float", [](SparseMatrixWrapper &wrap) {
        return wrap.is_float();
    }).def("transpose_", [](SparseMatrixWrapper &wrap) {
        wrap.perform([](auto &x){x.transpose();});
    }).def("emit", [](SparseMatrixWrapper &wrap, bool to_stdout) {
        auto func = [to_stdout](auto &x) {
            if(to_stdout) std::cout << x;
            else          std::cerr << x;
        };
        wrap.perform(func);
    }, py::arg("to_stdout")=false)
    .def("__str__", [](SparseMatrixWrapper &wrap) {
        std::string msg = ks::sprintf("Matrix of %zu/%zu elements of %s, %zu nonzeros", wrap.rows(), wrap.columns(), wrap.is_float() ? "float32": "double", wrap.nnz()).data();
        return msg;
    });
}
