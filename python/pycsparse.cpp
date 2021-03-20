#include "pycsparse.h"
#include "smw.h"

#if BUILD_CSR_CLUSTERING

py::object run_kmpp_noso(const PyCSparseMatrix &smw, py::object msr, py::int_ k, double gamma_beta, uint64_t seed, unsigned nkmc, unsigned ntimes,
                         py::ssize_t lspp, bool use_exponential_skips, py::ssize_t n_local_trials,
                         py::object weights) {
    return py_kmeanspp_noso(smw, msr, k, gamma_beta, seed, nkmc, ntimes, lspp, use_exponential_skips, n_local_trials, weights);
}
#endif

void init_pycsparse(py::module &m) {
#if BUILD_CSR_CLUSTERING
    py::class_<PyCSparseMatrix>(m, "CSparseMatrix").def(py::init<py::object>(), py::arg("sparray"))
    .def("__str__", [](const PyCSparseMatrix &x) {return std::string("CSparseMatrix, ") + std::to_string(x.rows()) + "x" + std::to_string(x.columns()) + ", " + std::to_string(x.nnz());})
    .def("columns", [](const PyCSparseMatrix &x) {return x.columns();})
    .def("rows", [](const PyCSparseMatrix &x) {return x.rows();})
    .def("nnz", [](const PyCSparseMatrix &x) {return x.nnz();})
    .def("rowsel", [](const PyCSparseMatrix &x, py::array idxsel) {
        auto inf = idxsel.request();
        int idxsel_type = inf.itemsize == 8 ? 2: inf.itemsize == 4 ? 1: inf.itemsize == 2 ? 3 : inf.itemsize == 1 ? 0: -1;
        if(idxsel_type < 0) throw std::runtime_error("idxseltype is not 1, 2, 4 or 8 bytes");

        int ip_fmt;
        switch(x.indptr_t_[0]) {
            case 'l': case 'L': ip_fmt = 2; break;
            case 'i': case 'I': ip_fmt = 1; break;
            case 'h': case 'H': ip_fmt = 3; break;
            default: throw std::invalid_argument("indptr_t not supported");
        }
        std::vector<double> vals;
        std::vector<uint64_t> idx;
        py::array_t<uint64_t> retip(inf.size + 1);
        auto retipptr = (uint64_t *)retip.request().ptr;
        retipptr[0] = 0;
        for(Py_ssize_t i = 0; i < inf.size; ++i) {
            uint64_t ind;
            if(idxsel_type == 1) {
                ind = ((uint32_t *)inf.ptr)[i];
            } else if(idxsel_type == 3) {
                ind = ((uint16_t *)inf.ptr)[i];
            } else if(idxsel_type == 2) {
                ind = ((uint64_t *)inf.ptr)[i];
            } else if(idxsel_type == 0) {
                ind = ((uint8_t *)inf.ptr)[i];
            } else throw std::runtime_error("idxsel_type invalid");
            size_t start, stop;
            if(ip_fmt == 2) {
                start = ((uint64_t *)x.indptrp_)[ind];
                stop = ((uint64_t *)x.indptrp_)[ind + 1];
            } else if(ip_fmt == 1) {
                start = ((uint32_t *)x.indptrp_)[ind];
                stop = ((uint32_t *)x.indptrp_)[ind + 1];
            } else {
                start = ((uint16_t *)x.indptrp_)[ind];
                stop = ((uint16_t *)x.indptrp_)[ind + 1];
            }
            size_t lastnnz = retipptr[i];
            retipptr[i + 1] = stop - start + retipptr[i];
            vals.resize(retipptr[i + 1]);
            idx.resize(retipptr[i + 1]);
            switch(std::tolower(x.indices_t_[0])) {
                case 'i': case 'I':
                    std::copy((uint32_t *)x.indicesp_ + start, (uint32_t *)x.indicesp_ + stop, idx.data() + lastnnz); break;
                case 'L': case 'l': case 'Q': case 'q':  std::copy((uint64_t *)x.indicesp_ + start, (uint64_t *)x.indicesp_ + stop, idx.data() + lastnnz); break;
                case 'h': case 'H':  std::copy((uint16_t *)x.indicesp_ + start, (uint16_t *)x.indicesp_ + stop, idx.data() + lastnnz); break;
                case 'b': case 'B':  std::copy((uint8_t *)x.indicesp_ + start, (uint8_t *)x.indicesp_ + stop, idx.data() + lastnnz); break;
                default: throw std::runtime_error(x.indices_t_ + " is unknown dtype");
            }
            switch(std::tolower(x.data_t_[0])) {
                case 'f': std::copy((float *)x.datap_ + start, (float *)x.datap_ + stop, vals.data() + lastnnz); break;
                case 'd': std::copy((double *)x.datap_ + start, (double *)x.datap_ + stop, vals.data() + lastnnz); break;
                case 'L': case 'l': std::copy((uint64_t *)x.datap_ + start, (uint64_t *)x.datap_ + stop, vals.data() + lastnnz); break;
                case 'I': case 'i': std::copy((uint32_t *)x.datap_ + start, (uint32_t *)x.datap_ + stop, vals.data() + lastnnz); break;
                case 'B': case 'b': std::copy((uint8_t *)x.datap_ + start, (uint8_t *)x.datap_ + stop, vals.data() + lastnnz); break;
                case 'H': case 'h': std::copy((uint16_t *)x.datap_ + start, (uint16_t *)x.datap_ + stop, vals.data() + lastnnz); break;
            }
        }
        py::array_t<double> retv(vals.size());
        std::copy(vals.begin(), vals.end(), (double *)retv.request().ptr);
        py::array_t<uint64_t> reti(idx.size());
        std::copy(idx.begin(), idx.end(), (uint64_t *)reti.request().ptr);
        return py::make_tuple(retv, reti, retip, py::make_tuple(py::int_(inf.size), py::int_(x.columns())));
    }).def_static("from_items", [](py::object data, py::object idx, py::object indptr, py::object shape) -> PyCSparseMatrix {
        auto da = py::cast<py::array>(data), ia = py::cast<py::array>(idx), ipa = py::cast<py::array>(indptr);
        auto sseq = py::cast<py::sequence>(shape);
        return PyCSparseMatrix(data, ia, ipa, sseq[0].cast<Py_ssize_t>(), sseq[1].cast<Py_ssize_t>(), da.request().size);
    }, py::arg("data"), py::arg("indices"), py::arg("indptr"), py::arg("shape"));

     m.def("kmeanspp", [](const PyCSparseMatrix &smw, const SumOpts &so, py::object weights) {
        return run_kmpp_noso(smw, py::int_(int(so.dis)), py::int_(int(so.k)),  so.gamma, so.seed, so.kmc2_rounds, std::max(int(so.extra_sample_tries) - 1, 0),
                       so.lspp, so.use_exponential_skips, so.n_local_trials, weights);
    },
    "Computes a selecion of points from the matrix pointed to by smw, returning indexes for selected centers, along with assignments and costs for each point.",
       py::arg("smw"),
       py::arg("opts"),
       py::arg("weights") = py::none()
    );
    m.def("kmeanspp",  run_kmpp_noso,
       "Computes a selecion of points from the matrix pointed to by smw, returning indexes for selected centers, along with assignments and costs for each point."
       "\nSet nkmc to -1 to perform streaming kmeans++ (kmc2 over the full dataset), which parallelizes better but may yield a lower-quality result.\n",
       py::arg("smw"), py::arg("msr"), py::arg("k"), py::arg("prior") = 0., py::arg("seed") = 0, py::arg("nkmc") = 0, py::arg("ntimes") = 1,
       py::arg("lspp") = 0, py::arg("expskips") = false, py::arg("n_local_trials") = 1,
       py::arg("weights") = py::none()
    );
    m.def("greedy_select",  [](PyCSparseMatrix &smw, const SumOpts &so) {
        std::vector<uint64_t> centers;
        const Py_ssize_t nr = smw.rows();
        std::vector<double> dret;
        smw.perform([&](auto &matrix) {
            std::tie(centers, dret) = m2greedysel(matrix, so);
        });
        auto dtstr = size2dtype(nr);
        auto ret = py::array(py::dtype(dtstr), std::vector<Py_ssize_t>{nr});
        py::array_t<double> costs(smw.rows());
        auto rpi = ret.request(), cpi = costs.request();
        std::copy(dret.begin(), dret.end(), (double *)cpi.ptr);
        switch(dtstr[0]) {
            case 'L': std::copy(centers.begin(), centers.end(), (uint64_t *)rpi.ptr); break;
            case 'I': std::copy(centers.begin(), centers.end(), (uint32_t *)rpi.ptr); break;
            case 'H': std::copy(centers.begin(), centers.end(), (uint16_t *)rpi.ptr); break;
            case 'B': std::copy(centers.begin(), centers.end(), (int8_t *)rpi.ptr); break;
        }
        return py::make_tuple(ret, costs);
    }, "Computes a greedy selection of points from the matrix pointed to by smw, returning indexes and a vector of costs for each point. To allow for outliers, use the outlier_fraction parameter of Sumopts.",
       py::arg("smw"), py::arg("sumopts"));
    m.def("d2_select",  [](const PyCSparseMatrix &smw, const SumOpts &so, py::object weights) {
        std::vector<uint32_t> centers, asn;
        std::vector<double> dc;
        double *wptr = nullptr;
        blz::DV<double> tmpw;
        if(py::isinstance<py::array>(weights)) {
            auto inf = py::cast<py::array>(weights).request();
            switch(inf.format.front()) {
                case 'd': wptr = (double *)inf.ptr; break;
                case 'f': tmpw = blz::make_cv((float *)inf.ptr, smw.rows()); wptr = tmpw.data(); break;
                default: throw std::invalid_argument("Wrong format weights");
            }
        }
        auto lhs = std::tie(centers, asn, dc);
        smw.perform([&](auto &x) {lhs = minicore::m2d2(x, so, wptr);});
        py::array_t<uint64_t> ret(centers.size());
        py::array_t<uint32_t> retasn(smw.rows());
        py::array_t<double> costs(smw.rows());
        auto rpi = ret.request(), api = retasn.request(), cpi = costs.request();
        std::copy(centers.begin(), centers.end(), (uint64_t *)rpi.ptr);
        std::copy(dc.begin(), dc.end(), (double *)cpi.ptr);
        std::copy(asn.begin(), asn.end(), (uint32_t *)api.ptr);
        return py::make_tuple(ret, retasn, costs);
    }, "Computes a selecion of points from the matrix pointed to by smw, returning indexes for selected centers, along with assignments and costs for each point.",
       py::arg("smw"), py::arg("sumopts"), py::arg("weights") = py::none());
#else
    std::fprintf(stderr, "[%s] PyCSparseMatrix bindings not created\n", __PRETTY_FUNCTION__);
#endif
}
