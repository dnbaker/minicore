#ifndef SMW_H
#define SMW_H
#include "pyfgc.h"
#include "blaze/util/Serialization.h"
#include "minicore/util/csc.h"


dist::DissimilarityMeasure assure_dm(py::object obj);

struct SparseMatrixWrapper {
    SparseMatrixWrapper(std::string path, bool use_float=true) {
        blaze::Archive<std::ifstream> arch(path);
        if(use_float) arch >> this->getfloat();
                else  arch >> this->getdouble();
    }
    void tofile(std::string path) {
        blaze::Archive<std::ofstream> arch(path);
        perform([&arch](auto &x) {arch << x;});
    }
private:
    template<typename IndPtrT, typename IndicesT, typename Data>
    SparseMatrixWrapper(IndPtrT *indptr, IndicesT *indices, Data *data,
                  size_t nnz, uint32_t nfeat, uint32_t nitems, bool skip_empty=false, bool use_float=true) {
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
public:
    template<typename FT>
    SparseMatrixWrapper(blz::SM<FT> &&mat): matrix_(std::move(mat)) {}
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
    template<typename IpT, typename IdxT, typename DataT>
    SparseMatrixWrapper(IpT *indptr, IdxT *idx, DataT *data, size_t xdim, size_t ydim, size_t nnz, bool use_float=true, bool skip_empty=true) {
        if(use_float)
            matrix_ = csc2sparse<float>(CSCMatrixView<IpT, IdxT, DataT>(indptr, idx, data, nnz, ydim, xdim));
        else
            matrix_ = csc2sparse<double>(CSCMatrixView<IpT, IdxT, DataT>(indptr, idx, data, nnz, ydim, xdim));
    }
    SparseMatrixWrapper(py::object spmat, py::object skip_empty_py, py::object use_float_py) {
        py::array indices = spmat.attr("indices"), indptr = spmat.attr("indptr"), data = spmat.attr("data");
        py::tuple shape = py::cast<py::tuple>(spmat.attr("shape"));
        const bool use_float = py::cast<bool>(use_float_py), skip_empty = py::cast<bool>(skip_empty_py);
        size_t xdim = py::cast<size_t>(shape[0]), ydim = py::cast<size_t>(shape[1]);
        size_t nnz = py::cast<size_t>(spmat.attr("nnz"));
        auto indbuf = indices.request(), indpbuf = indptr.request(), databuf = data.request();
        void *datptr = databuf.ptr, *indptrptr = indpbuf.ptr, *indicesptr = indbuf.ptr;

#define __DISPATCH(T1, T2, T3) do { \
        if(use_float) {\
            if(databuf.readonly || indbuf.readonly) {\
                DBG_ONLY(std::fprintf(stderr, "Read only floats\n");) \
                matrix_ = csc2sparse<float>(CSCMatrixView<T1, const T2, const T3>(reinterpret_cast<T1 *>(indptrptr), reinterpret_cast<const T2 *>(const_cast<const void *>(indicesptr)), reinterpret_cast<const T3 *>(const_cast<const void *>(datptr)), nnz, ydim, xdim), skip_empty); \
            } else { \
                DBG_ONLY(std::fprintf(stderr, "Const reading of floats\n");) \
                matrix_ = csc2sparse<float>(CSCMatrixView<T1, T2, T3>(reinterpret_cast<T1 *>(indptrptr), reinterpret_cast<T2 *>(indicesptr), reinterpret_cast<T3 *>(datptr), nnz, ydim, xdim), skip_empty); \
            }\
        } else { \
            if(databuf.readonly || indbuf.readonly) {\
                DBG_ONLY(std::fprintf(stderr, "Read only reading of doubles\n");) \
                matrix_ = csc2sparse<double>(CSCMatrixView<T1, const T2, const T3>(reinterpret_cast<T1 *>(indptrptr), reinterpret_cast<const T2 *>(const_cast<const void *>(indicesptr)), reinterpret_cast<const T3 *>(const_cast<const void *>(datptr)), nnz, ydim, xdim), skip_empty); \
            } else { \
                DBG_ONLY(std::fprintf(stderr, "Const reading of doubles\n");) \
                matrix_ = csc2sparse<double>(CSCMatrixView<T1, T2, T3>(reinterpret_cast<T1 *>(indptrptr), reinterpret_cast<T2 *>(indicesptr), reinterpret_cast<T3 *>(datptr), nnz, ydim, xdim), skip_empty); \
            }\
        }\
        return; \
    } while(0)
#define __DISPATCH_IF(T1, T2, T3) do { \
                if(py::format_descriptor<T3>::format() == databuf.format) { \
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
    std::vector<std::pair<uint32_t, double>> row2tups(size_t r) const {
        if(r > rows()) throw std::invalid_argument("Cannot get tuples from a row that dne");
        std::vector<std::pair<uint32_t, double>> ret;
        perform([&ret,r](const auto &x) {
            for(const auto &pair: row(x, r))
                ret.emplace_back(pair.index(), pair.value());
        });
        return ret;
    }
    std::pair<void *, bool> get_opaque() {
        return {is_float() ? static_cast<void *>(&std::get<SMF>(matrix_)): static_cast<void *>(&std::get<SMD>(matrix_)),
                is_float()};
    }
};

template<typename Mat>
inline py::tuple py_kmeanspp(const Mat &smw, py::object msr, Py_ssize_t k, double gamma_beta, uint64_t seed, unsigned nkmc, unsigned ntimes,
                 int lspp,
                 int use_exponential_skips,
                 py::object weights)
{
    const void *wptr = nullptr;
    int kind = -1;
    const auto mmsr = assure_dm(msr);
    const size_t nr = smw.rows();
    std::fprintf(stderr, "Performing kmeans++ with msr %d/%s\n", (int)mmsr, cmp::msr2str(mmsr));
    if(py::isinstance<py::array>(weights)) {
        auto arr = py::cast<py::array>(weights);
        auto info = arr.request();
        if(info.format.size() > 1) throw std::invalid_argument(std::string("Invalid array format: ") + info.format);
        switch(info.format.front()) {
            case 'l': case 'i': case 'I': case 'H': case 'L': case 'B': case 'f': case 'd': kind = info.format.front(); break;
            default:throw std::invalid_argument(std::string("Invalid array format: ") + info.format + ". Expected 'd', 'f', 'i', 'I', 'l', 'L', 'B', 'h'.\n");
        }
        wptr = info.ptr;
    }
    std::fprintf(stderr, "ki: %d\n", int(k));
    wy::WyRand<uint64_t> rng(seed);
    const auto psum = gamma_beta * smw.columns();
    const blz::StaticVector<double, 1> prior({gamma_beta});
    auto cmp = [measure=mmsr, psum,&prior](const auto &x, const auto &y) {
        // Note that this has been transposed
        return cmp::msr_with_prior(measure, y, x, prior, psum, sum(y), sum(x));
    };
    py::array_t<uint32_t> ret(k);
    int retasnbits;
    if(k <= 256) {
        retasnbits = 8;
        std::fprintf(stderr, "uint8 labels\n");
    } else if(k <= 63356) {
        retasnbits = 16;
        std::fprintf(stderr, "uint16 labels\n");
    } else if(k <= 0xFFFFFFFF) {
        retasnbits = 32;
        std::fprintf(stderr, "uint32 labels\n");
    } else {
        retasnbits = 64;
        std::fprintf(stderr, "uint64 labels. >4 billion centers, are you crazy?\n");
    }
    const char *kindstr = retasnbits == 8 ? "B": retasnbits == 16 ? "H": retasnbits == 32 ? "U": "L";
    py::array retasn(py::dtype(kindstr), std::vector<Py_ssize_t>{{Py_ssize_t(nr)}});
    auto retai = retasn.request();
    auto rptr = (uint32_t *)ret.request().ptr;
    py::array_t<float> costs(smw.rows());
    auto costp = (float *)costs.request().ptr;
    try {
    smw.perform([&](auto &x) {
        //using RT = decltype(repeatedly_get_initial_centers(x, rng, k, nkmc, ntimes, cmp));
        auto sol =
            kind == -1 ?
            repeatedly_get_initial_centers(x, rng, k, nkmc, ntimes, lspp, use_exponential_skips, cmp)
            : kind == 'f' ? repeatedly_get_initial_centers(x, rng, k, nkmc, ntimes, lspp, use_exponential_skips, cmp, (const float *)wptr)
            : kind == 'd' ? repeatedly_get_initial_centers(x, rng, k, nkmc, ntimes, lspp, use_exponential_skips, cmp, (const double *)wptr)
            : kind == 'I' ? repeatedly_get_initial_centers(x, rng, k, nkmc, ntimes, lspp, use_exponential_skips, cmp, (const unsigned *)wptr)
            : kind == 'i' ? repeatedly_get_initial_centers(x, rng, k, nkmc, ntimes, lspp, use_exponential_skips, cmp, (const int *)wptr)
            : kind == 'H' ? repeatedly_get_initial_centers(x, rng, k, nkmc, ntimes, lspp, use_exponential_skips, cmp, (const uint16_t *)wptr)
            : kind == 'L' ? repeatedly_get_initial_centers(x, rng, k, nkmc, ntimes, lspp, use_exponential_skips, cmp, (const uint64_t *)wptr)
            : kind == 'l' ? repeatedly_get_initial_centers(x, rng, k, nkmc, ntimes, lspp, use_exponential_skips, cmp, (const int64_t *)wptr)
            : kind == 'B' ? repeatedly_get_initial_centers(x, rng, k, nkmc, ntimes, lspp, use_exponential_skips, cmp, (const uint8_t *)wptr)
            : repeatedly_get_initial_centers(x, rng, k, nkmc, ntimes, lspp, use_exponential_skips, cmp, (const int *)wptr);
        auto &[lidx, lasn, lcosts] = sol;
        for(size_t i = 0; i < lidx.size(); ++i) {
            std::fprintf(stderr, "selected point %u for center %zu\n", lidx[i], i);
            if(lidx[i] > nr) std::fprintf(stderr, "Warning: 'center' id is > # centers\n");
        }
        for(size_t i = 0; i < lasn.size(); ++i) {
            if(lasn[i] > nr) {
                std::fprintf(stderr, "asn %zu is %u (> nr)\n", i, unsigned(lasn[i]));
            }
        }
        assert(lidx.size() == ki);
        assert(lasn.size() == smw.rows());
        switch(retasnbits) {
            case 8: {
                auto raptr = (uint8_t *)retai.ptr;
                OMP_PFOR
                for(size_t i = 0; i < lasn.size(); ++i)
                    raptr[i] = lasn[i];
            } break;
            case 16: {
                auto raptr = (uint16_t *)retai.ptr;
                OMP_PFOR
                for(size_t i = 0; i < lasn.size(); ++i)
                    raptr[i] = lasn[i];
            } break;
            case 32: {
                auto raptr = (uint32_t *)retai.ptr;
                OMP_PFOR
                for(size_t i = 0; i < lasn.size(); ++i)
                    raptr[i] = lasn[i];
            } break;
            case 64: {
                auto raptr = (uint64_t *)retai.ptr;
                OMP_PFOR
                for(size_t i = 0; i < lasn.size(); ++i)
                    raptr[i] = lasn[i];
            } break;
            default: __builtin_unreachable();
        }
        OMP_PFOR
        for(size_t i = 0; i < lcosts.size(); ++i)
            costp[i] = lcosts[i];
        for(size_t i = 0; i < lidx.size(); ++i)
            rptr[i] = lidx[i];
    });
    } catch(const TODOError &) {
        throw std::invalid_argument("Unsupported measure");
    } catch(const std::runtime_error &ex) {
        throw static_cast<std::exception>(ex);
    } catch(...) {throw std::invalid_argument("No idea what exception was thrown, but it was unrecoverable.");}
    return py::make_tuple(ret, retasn, costs);
}

template<typename Mat>
inline py::tuple py_kmeanspp_so(const Mat &smw, const SumOpts &sm, py::object weights) {
    return py_kmeanspp(smw, py::int_((int)sm.dis), sm.k, sm.gamma, sm.seed, sm.kmc2_rounds, std::max(sm.extra_sample_tries - 1, 0u),
                       sm.lspp, sm.use_exponential_skips, weights);
}
template<typename Mat>
py::object py_kmeanspp_noso(Mat &smw, py::object msr, py::int_ k, double gamma_beta, uint64_t seed, unsigned nkmc, unsigned ntimes,
                          Py_ssize_t lspp, bool use_exponential_skips,
                          py::object weights)
    {
        if(gamma_beta < 0.) {
            gamma_beta = 1. / smw.columns();
            std::fprintf(stderr, "Warning: unset beta prior defaults to 1 / # columns (%g)\n", gamma_beta);
        } else if(gamma_beta == 0.) {
            std::fprintf(stderr, "Note: beta prior set to 0; this may yield nans in some cases\n");
        }
        const void *wptr = nullptr;
        int kind = -1;
        const auto mmsr = assure_dm(msr);
        const size_t nr = smw.rows();
        std::fprintf(stderr, "Performing kmeans++ with msr %d/%s\n", (int)mmsr, cmp::msr2str(mmsr));
        if(py::isinstance<py::array>(weights)) {
            auto arr = py::cast<py::array>(weights);
            auto info = arr.request();
            if(info.format.size() > 1) throw std::invalid_argument(std::string("Invalid array format: ") + info.format);
            switch(info.format.front()) {
                case 'i': case 'u': case 'f': case 'd': kind = info.format.front(); break;
                default:throw std::invalid_argument(std::string("Invalid array format: ") + info.format + ". Expected 'd', 'f', 'i', or 'u'.\n");
            }
            wptr = info.ptr;
        }
        auto ki = k.cast<Py_ssize_t>();
        wy::WyRand<uint64_t> rng(seed);
        const auto psum = gamma_beta * smw.columns();
        const blz::StaticVector<double, 1> prior({gamma_beta});
        auto cmp = [measure=mmsr, psum,&prior](const auto &x, const auto &y) {
            // Note that this has been transposed
            return cmp::msr_with_prior(measure, y, x, prior, psum, sum(y), sum(x));
        };
        py::array_t<uint32_t> ret(ki);
        py::object retasn = py::none();
        int retasnbits;
        if(ki <= 256) {
            retasn = py::array_t<uint8_t>(nr);
            retasnbits = 8;
        } else if(ki <= 63356) {
            retasn = py::array_t<uint16_t>(nr);
            retasnbits = 16;
        } else {
            retasn = py::array_t<uint32_t>(nr);
            retasnbits = 32;
        }
        auto retai = py::cast<py::array>(retasn).request();
        auto rptr = (uint32_t *)ret.request().ptr;
        py::array_t<float> costs(smw.rows());
        auto costp = (float *)costs.request().ptr;
        smw.perform([&](auto &x) {
            //using RT = decltype(repeatedly_get_initial_centers(x, rng, ki, nkmc, ntimes, cmp));
            auto sol =
                kind == -1 ?
                repeatedly_get_initial_centers(x, rng, ki, nkmc, ntimes, lspp, use_exponential_skips, cmp)
                : kind == 'f' ? repeatedly_get_initial_centers(x, rng, ki, nkmc, ntimes, lspp, use_exponential_skips, cmp, (const float *)wptr)
                : kind == 'd' ? repeatedly_get_initial_centers(x, rng, ki, nkmc, ntimes, lspp, use_exponential_skips, cmp, (const double *)wptr)
                : kind == 'u' ? repeatedly_get_initial_centers(x, rng, ki, nkmc, ntimes, lspp, use_exponential_skips, cmp, (const unsigned *)wptr)
                : repeatedly_get_initial_centers(x, rng, ki, nkmc, ntimes, lspp, use_exponential_skips, cmp, (const int *)wptr);
            auto &[lidx, lasn, lcosts] = sol;
            for(size_t i = 0; i < lidx.size(); ++i) {
                std::fprintf(stderr, "selected point %u for center %zu\n", lidx[i], i);
                if(lidx[i] > nr) std::fprintf(stderr, "Warning: 'center' id is > # centers\n");
            }
            for(size_t i = 0; i < lasn.size(); ++i) {
                if(lasn[i] > nr) {
                    std::fprintf(stderr, "asn %zu is %u (> nr)\n", i, unsigned(lasn[i]));
                }
            }
            assert(lidx.size() == ki);
            assert(lasn.size() == smw.rows());
            switch(retasnbits) {
                case 8: {
                    auto raptr = (uint8_t *)retai.ptr;
                    OMP_PFOR
                    for(size_t i = 0; i < lasn.size(); ++i)
                        raptr[i] = lasn[i];
                } break;
                case 16: {
                    auto raptr = (uint16_t *)retai.ptr;
                    OMP_PFOR
                    for(size_t i = 0; i < lasn.size(); ++i)
                        raptr[i] = lasn[i];
                } break;
                case 32: {
                    auto raptr = (uint32_t *)retai.ptr;
                    OMP_PFOR
                    for(size_t i = 0; i < lasn.size(); ++i)
                        raptr[i] = lasn[i];
                } break;
                default: __builtin_unreachable();
            }
            OMP_PFOR
            for(size_t i = 0; i < lcosts.size(); ++i)
                costp[i] = lcosts[i];
            for(size_t i = 0; i < lidx.size(); ++i)
                rptr[i] = lidx[i];
        });
        return py::make_tuple(ret, retasn, costs);
    }


dist::DissimilarityMeasure assure_dm(py::object obj);

#endif
