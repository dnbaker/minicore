#include "smw.h"
#include "pyfgc.h"

template<typename IT1, typename IT2, typename FT>
Py_ssize_t filtered_nonzeros(py::handle matrix, const std::vector<Py_ssize_t> &indices) {
    auto rows = py::cast<py::array_t<IT1>>(matrix.attr("row"));
    auto cols = py::cast<py::array_t<IT2>>(matrix.attr("col"));
    auto data = py::cast<py::array_t<FT>>(matrix.attr("data"));
    auto colp = (IT1 *)cols.request().ptr;
    auto datap = (FT *)data.request().ptr;
    const auto nnz = py::cast<Py_ssize_t>(matrix.attr("nnz"));
    Py_ssize_t ret = 0;
    for(Py_ssize_t i = 0; i < nnz; ++i) {
        auto cid = colp[i];
        if(indices[cid] >= 0 and datap[i] >= 0.) {
            ++ret; 
        }
    }
    return ret;
}

template<typename IT1, typename IT2, typename FT>
auto getpointers(py::handle matrix) {
    auto rows = py::cast<py::array_t<IT1>>(matrix.attr("row"));
    auto cols = py::cast<py::array_t<IT2>>(matrix.attr("col"));
    auto data = py::cast<py::array_t<FT>>(matrix.attr("data"));
    auto colp = (IT1 *)cols.request().ptr;
    auto rowp = (IT2 *)rows.request().ptr;
    auto datap = (FT *)data.request().ptr;
    return std::make_tuple(rowp, colp, datap);
}


template <typename IT1, typename IT2, typename FT1, typename IT3, typename IT4, typename FT2>
void update_all(const IT1 *srcrow, const IT2 *srccol, const FT1 *srcdat, IT3 *destrow, IT4 *destcol, FT2 *destdat,
                size_t nzi, size_t mynnz, const std::vector<Py_ssize_t> &indices) {
    for(size_t i = 0; i < mynnz; ++i) {
        auto f = indices[srccol[i]];
        auto v = srcdat[i];
        if(f >= Py_ssize_t(0) && v >= FT1(0)) {
            destdat[nzi] = srcdat[i];
            destrow[nzi] = srcrow[i];
            destcol[nzi] = f;
            ++nzi;
        }
    }
}


template<typename IT, typename T>
void update_all(py::handle mat, IT *row, IT *col, T *data, size_t offset, size_t nzi, size_t nnz, const std::vector<Py_ssize_t> &indices) {
#define DOUPDATE(T1, T2, T3) \
    try { \
        auto [srcrow, srccol, srcdata] = getpointers<T1, T2, T3>(mat); \
        update_all(srcrow, srccol, srcdata, row, col, data, nzi, nnz, indices); \
        return;\
    } catch(...) {}
    DOUPDATE(uint32_t, uint32_t, double);
    DOUPDATE(int32_t, int32_t, double);
    DOUPDATE(uint32_t, int32_t, double);
    DOUPDATE(int32_t, uint32_t, double);
    DOUPDATE(uint64_t, uint64_t, double);
    DOUPDATE(int64_t, int64_t, double);
    DOUPDATE(uint64_t, int64_t, double);
    DOUPDATE(int64_t, uint64_t, double);
    DOUPDATE(uint32_t, uint32_t, double);
    DOUPDATE(int32_t, int32_t, double);
    DOUPDATE(uint32_t, int32_t, double);
    DOUPDATE(int32_t, uint32_t, double);
    DOUPDATE(uint64_t, uint64_t, double);
    DOUPDATE(int64_t, int64_t, double);
    DOUPDATE(uint64_t, int64_t, double);
    DOUPDATE(int64_t, uint64_t, double);
    throw std::runtime_error("Failed to find a matching type");
}

Py_ssize_t filtered_nonzeros(py::handle matrix, const std::vector<Py_ssize_t> &indices) {
#define DOFNZ(T1, T2, T3) do {\
    try { \
        return filtered_nonzeros<T1, T2, T3>(matrix, indices); \
    } catch(...) {} \
    } while(0)
    DOFNZ(uint32_t, uint32_t, double);
    DOFNZ(uint32_t, uint32_t, float);
    DOFNZ(int32_t, int32_t, double);
    DOFNZ(int32_t, int32_t, float);
    DOFNZ(uint64_t, uint64_t, double);
    DOFNZ(uint64_t, uint64_t, float);
    DOFNZ(int64_t, int64_t, double);
    DOFNZ(int64_t, int64_t, float);
    throw std::runtime_error("Failed to find a matching type");
}

void init_merge(py::module &m) {
    m.def("merge", [](py::list matrices, py::list featmaps, py::list features) {
        assert(matrices.size() == featmaps.size());
        const Py_ssize_t nf = features.size();
        size_t nr = 0;
        std::vector<std::vector<Py_ssize_t>> luts;
        for(auto fm: featmaps) {
            nr += py::cast<Py_ssize_t>(fm.attr("n"));
            py::dict cvt = fm.attr("cvt");
            Py_ssize_t features_used = len(cvt);
            auto &lut = luts.emplace_back(nf, Py_ssize_t(-1));
            for(auto item: cvt) {
                lut[py::cast<Py_ssize_t>(item.first)] = py::cast<Py_ssize_t>(item.second);
            }
        }
        py::array_t<Py_ssize_t> rows, cols;
        py::array_t<double> data;
        py::array_t<uint64_t> shape(2);
        {
            auto sp = (uint64_t *)shape.request().ptr;
            sp[0] = nr; sp[1] = nf;
        }
        std::vector<Py_ssize_t> nnzs;
        size_t i = 0;
        for(auto mat: matrices) {
            nnzs.push_back(filtered_nonzeros(mat, luts[i++]));
        }
        Py_ssize_t nnz = std::accumulate(nnzs.begin(), nnzs.end(), Py_ssize_t(0));
        std::fprintf(stderr, "total of %zd nonzeros\n", nnz);
        rows.resize({nnz}); cols.resize({nnz}); data.resize({nnz});
        auto rowp = (Py_ssize_t *)rows.request().ptr, colp = (Py_ssize_t *)cols.request().ptr;
        auto datap = (double *)data.request().ptr;
        size_t offset = 0;
        size_t nzi = 0;
        auto lutit = luts.begin();
        auto nnzit = nnzs.begin();
        for(auto mat: matrices) {
            auto &lut = *lutit++;
            auto mynnz = *nnzit++;
            std::fprintf(stderr, "matrix %zd. %zd->%zd\n", lutit - luts.begin() + 1, nzi, nzi + mynnz);
            update_all(mat, rowp, colp, datap, offset, nzi, mynnz, lut);
            nzi += mynnz;
        }
        return std::make_tuple(rows, cols, data, shape);
    });
}
