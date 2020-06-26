#include "smw.h"
#include "pyfgc.h"


template<typename K>
auto getattr(py::handle src, const K &key) {
    if(!hasattr(src, key)) {
        throw std::runtime_error(std::string("Missing key ") + std::string(key) + ". Are you sure your matrix is in scipy.sparse.COOMatrix format?");
    }
    return src.attr(key);
}

template<typename IT1, typename IT2, typename FT>
Py_ssize_t filtered_nonzeros(py::handle matrix, const std::vector<Py_ssize_t> &indices) {
    auto cols = py::cast<py::array_t<IT2>>(getattr(matrix, "col"));
    auto data = py::cast<py::array_t<FT>>(getattr(matrix, "data"));
    auto colp = (IT1 *)cols.request().ptr;
    auto datap = (FT *)data.request().ptr;
    const auto nnz = py::cast<Py_ssize_t>(getattr(matrix, "nnz"));
    Py_ssize_t ret = 0;
    for(Py_ssize_t i = 0; i < nnz; ++i) {
        auto cid = colp[i];
        if(indices.at(cid) >= 0 and datap[i] >= 0.) {
            ++ret; 
        }
    }
    return ret;
}

template<typename IT1, typename IT2, typename FT>
auto getpointers(py::handle matrix) {
    if(!hasattr(matrix, "row") || !hasattr(matrix, "col") || !hasattr(matrix, "data")) {
        throw std::logic_error("Cannot get COO pointers from a non-COO matrix");
    }
    auto rows = py::cast<py::array_t<IT1>>(getattr(matrix, "row"));
    auto cols = py::cast<py::array_t<IT2>>(getattr(matrix, "col"));
    auto data = py::cast<py::array_t<FT>>(getattr(matrix, "data"));
    auto colp = (IT1 *)cols.request().ptr;
    if(!colp) throw std::runtime_error("colp is null!");
    auto rowp = (IT2 *)rows.request().ptr;
    if(!colp) throw std::runtime_error("rowp is null!");
    auto datab = data.request();
    auto datap = (FT *)datab.ptr;
    if(!colp) throw std::runtime_error("datap is null!");
    VERBOSE_ONLY(std::fprintf(stderr, "[%s] Succeeded in getting pointers, %p, %p, %p", __PRETTY_FUNCTION__, (void *)rowp, (void *)colp, (void *)datap);)
    return std::make_tuple(rowp, colp, datap, datab.size);
}


template <typename IT1, typename IT2, typename FT1>
Py_ssize_t count_nonzeros(const IT1 *srcrow, const IT2 *srccol, const FT1 *srcdat,
                          Py_ssize_t ncoo, const std::vector<Py_ssize_t> &indices) {
    Py_ssize_t ret = 0;
    for(Py_ssize_t i = 0; i < ncoo; ++i) {
        std::cerr << "Getting idx " << i << '\n';
        auto idx = srccol[i];
        std::fprintf(stderr, "Getting ii (%p + %zu) (integer size: %zu)\n", (void *)indices.data(), idx, sizeof(IT2));
        auto ii = indices.at(idx);
        auto v = srcdat[i];
        std::fprintf(stderr, "ii: %zd. value there: %g\n", ii, double(v));
        if(ii >= 0) {
            std::fprintf(stderr, "ii (%zd) > 0\n", ii);
            ++ret;
        }
    }
    return ret;
}

Py_ssize_t count_nonzeros(py::object mat, Py_ssize_t ncoo, const std::vector<Py_ssize_t> &indices) {
#define DOUPDATE(T1, T2, T3) \
    try { \
        auto [srcrow, srccol, srcdata, ncoo] = getpointers<T1, T2, T3>(mat); \
        return count_nonzeros(srcrow, srccol, srcdata, ncoo, indices); \
    } catch(...) {}
    DOUPDATE(uint32_t, uint32_t, double);
    DOUPDATE(int32_t, int32_t, double);
    //DOUPDATE(uint32_t, int32_t, double);
    //DOUPDATE(int32_t, uint32_t, double);
    DOUPDATE(uint64_t, uint64_t, double);
    DOUPDATE(int64_t, int64_t, double);
    //DOUPDATE(uint64_t, int64_t, double);
    //DOUPDATE(int64_t, uint64_t, double);
    DOUPDATE(uint32_t, uint32_t, double);
    DOUPDATE(int32_t, int32_t, double);
    //DOUPDATE(uint32_t, int32_t, double);
    //DOUPDATE(int32_t, uint32_t, double);
    DOUPDATE(uint64_t, uint64_t, double);
    DOUPDATE(int64_t, int64_t, double);
    //DOUPDATE(uint64_t, int64_t, double);
    //DOUPDATE(int64_t, uint64_t, double);
    throw std::runtime_error("Failed to find a matching type");
#undef DOUPDATE
}

template <typename IT1, typename IT2, typename FT1, typename IT3, typename IT4, typename FT2>
Py_ssize_t update_all(const IT1 *srcrow, const IT2 *srccol, const FT1 *srcdat, IT3 *destrow, IT4 *destcol, FT2 *destdat, size_t nzi,
                Py_ssize_t ncoo, const std::vector<Py_ssize_t> &indices) {
    Py_ssize_t ret = 0;
#if 0
    destdat += nzi;
    destrow += nzi;
    destcol += nzi;
#endif
#ifndef NDEBUG
        if(destrow % sizeof(*destrow)) throw std::runtime_error(std::string("Unaligned destrow in ") + __PRETTY_FUNCTION__);
        if(destcol % sizeof(*destcol)) throw std::runtime_error(std::string("Unaligned destcol in ") + __PRETTY_FUNCTION__);
        if(destdat % sizeof(*destdat)) throw std::runtime_error(std::string("Unaligned destdat in ") + __PRETTY_FUNCTION__);
#endif
    for(Py_ssize_t i = 0; i < ncoo; ++i) {
        std::fprintf(stderr, "[%s] %zd/%zd. Input triplet: (%d, %d, %g)\n",
                     __PRETTY_FUNCTION__, i + 1, ncoo, int(srcrow[i]), int(srccol[i]), double(srcdat[i]));
#ifndef NDEBUG
        if(srcrow % sizeof(*srcrow)) throw std::runtime_error(std::string("Unaligned srcrow in ") + __PRETTY_FUNCTION__);
        if(srccol % sizeof(*srccol)) throw std::runtime_error(std::string("Unaligned srccol in ") + __PRETTY_FUNCTION__);
        if(srcdat % sizeof(*srcdat)) throw std::runtime_error(std::string("Unaligned srcdat in ") + __PRETTY_FUNCTION__);
#endif
        auto srcid = srccol[i];
        assert(srcid < indices.size());
        auto f = indices[srcid];
        if(f < 0) {
            std::fprintf(stderr, "skipping feature at srcid %d/%zd\n", int(srcid), indices[srcid]);
            continue;
        }
        std::fprintf(stderr, "destcol %d from  %d\n", int(f), int(srcid));
        auto v = srcdat[i];
        if(v >= static_cast<FT1>(0)) {
            std::fprintf(stderr, "Remapping original %d to %d, %zd so far\n", int(srcrow[i]), int(f), ret + 1);
            destdat[nzi] = v;
            destrow[nzi] = srcrow[i];
            destcol[nzi] = f;
            ++nzi;
            ++ret;
        } else {
            std::fprintf(stderr, "v %g was negative\n", double(v));
        }
    }
    return ret;
}


template<typename IT, typename T>
Py_ssize_t update_all(py::handle mat, IT *row, IT *col, T *data, size_t offset, size_t nzi, const std::vector<Py_ssize_t> &indices) {
#define DOUPDATE(T1, T2, T3) \
    try { \
        auto [srcrow, srccol, srcdata, ncoo] = getpointers<T1, T2, T3>(mat); \
        return update_all(srcrow, srccol, srcdata, row, col, data, nzi, ncoo, indices); \
    } catch(...) {}
    DOUPDATE(Py_ssize_t, Py_ssize_t, double);
    //DOUPDATE(int32_t, int32_t, double);
    //DOUPDATE(uint32_t, int32_t, double);
    //DOUPDATE(int32_t, uint32_t, double);
    //DOUPDATE(uint64_t, uint64_t, double);
    //DOUPDATE(int64_t, int64_t, double);
    //DOUPDATE(uint64_t, int64_t, double);
    //DOUPDATE(int64_t, uint64_t, double);
    //DOUPDATE(uint32_t, uint32_t, double);
    //DOUPDATE(int32_t, int32_t, double);
    //DOUPDATE(uint32_t, int32_t, double);
    //DOUPDATE(int32_t, uint32_t, double);
    //DOUPDATE(uint64_t, uint64_t, double);
    //DOUPDATE(int64_t, int64_t, double);
    //DOUPDATE(uint64_t, int64_t, double);
    //DOUPDATE(int64_t, uint64_t, double);
    throw std::runtime_error("Failed to find a matching type");
#undef DOUPDATE
}

Py_ssize_t filtered_nonzeros(py::handle matrix, const std::vector<Py_ssize_t> &indices) {
#define DOFNZ(T1, T2, T3) do {\
    try { \
        return filtered_nonzeros<T1, T2, T3>(matrix, indices); \
    } catch(...) {} \
    } while(0)
    DOFNZ(uint32_t, uint32_t, double);
    //DOFNZ(uint32_t, uint32_t, float);
    //DOFNZ(int32_t, int32_t, double);
    //DOFNZ(int32_t, int32_t, float);
    //DOFNZ(uint64_t, uint64_t, double);
    //DOFNZ(uint64_t, uint64_t, float);
    //DOFNZ(int64_t, int64_t, double);
    //DOFNZ(int64_t, int64_t, float);
#undef DOFNZ
    throw std::runtime_error("Failed to find a matching type");
}

void init_merge(py::module &m) {
    m.def("merge", [](py::list matrices, py::list featmaps, py::list features) {
        assert(matrices.size() == featmaps.size());
        const Py_ssize_t nf = features.size();
        std::vector<std::vector<Py_ssize_t>> luts;
        for(auto fm: featmaps) {
            py::dict cvt = fm.attr("cvt");
            auto &lut = luts.emplace_back(nf, Py_ssize_t(-1));
            for(auto item: cvt) {
                const Py_ssize_t key = py::cast<Py_ssize_t>(item.first), v = py::cast<Py_ssize_t>(item.second);
                VERBOSE_ONLY(std::fprintf(stderr, "K, V: %zd, %zd\n", key, v);)
                lut[key] = v;
            }
        }
        auto nmat = matrices.size();
        size_t nr = 0;
        Py_ssize_t max_nnz = 0;
        std::vector<Py_ssize_t> nnzs;
        for(size_t i = 0; i < nmat; ++i) {
            auto mat = matrices[i];
            auto shape = py::cast<py::tuple>(mat.attr("shape"));
            auto mynr = py::cast<Py_ssize_t>(shape[0]);
            auto mynnz = count_nonzeros(mat, py::cast<Py_ssize_t>(mat.attr("nnz")), luts[i]);
            VERBOSE_ONLY(std::fprintf(stderr, "Increment of %zu to %zu\n", mynr, nr);)
            nr += mynr;
            max_nnz += mynnz;
            nnzs.push_back(mynnz);
        }
        py::array_t<uint64_t> shape(2);
        {
            auto sp = (uint64_t *)shape.request().ptr;
            sp[0] = nr; sp[1] = nf;
        }
        VERBOSE_ONLY(std::fprintf(stderr, "maximum total of %zd nonzeros\n", max_nnz);)
        py::array_t<Py_ssize_t> rows, cols;
        py::array_t<double> data;
        rows.resize({max_nnz}); cols.resize({max_nnz}); data.resize({max_nnz});
        auto rowp = (Py_ssize_t *)rows.request().ptr, colp = (Py_ssize_t *)cols.request().ptr;
        auto datap = (double *)data.request().ptr;
        size_t offset = 0;
        Py_ssize_t nzi = 0;
        auto lutit = luts.begin();
        for(size_t i = 0; i < nmat; ++i) {
            auto mat = matrices[i];
            auto &lut = *lutit++;
            auto mynnz = update_all(mat, rowp, colp, datap, offset, nzi, lut);
            nzi += mynnz;
            std::fprintf(stderr, "%zu found %zd nnz \n", i, mynnz);
            if(mynnz != nnzs[i]) {
                std::fprintf(stderr, "nnz found earlier: %zu. nnz found now: %zd\n", nnzs[i], mynnz);
            }
        }
        std::fprintf(stderr, "total nzi: %zu. compared to maximum possible %zu\n", nzi, max_nnz);
        if(nzi != max_nnz) {
            throw std::runtime_error("Incorrect matrix sizes");
            rows.resize({nzi});
            cols.resize({nzi});
            data.resize({nzi});
        }
        return std::make_tuple(rows, cols, data, shape);
    }, py::arg("matrices"), py::arg("featmaps"), py::arg("features"));
}
