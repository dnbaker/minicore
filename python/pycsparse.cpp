#include "pycsparse.h"
#include "smw.h"

#if BUILD_CSR_CLUSTERING

py::object run_kmpp_noso(const PyCSparseMatrix &smw, py::object msr, py::int_ k, double gamma_beta, uint64_t seed, unsigned nkmc, unsigned ntimes,
                         Py_ssize_t lspp, bool use_exponential_skips,
                         py::object weights) {
    return py_kmeanspp_noso(smw, msr, k, gamma_beta, seed, nkmc, ntimes, lspp, use_exponential_skips, weights);
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
        const bool idx_is_4byte = inf.itemsize == 4;

        const bool ip_is_64 = std::tolower(x.indptr_t_[0]) == 'l';
        std::vector<double> vals;
        std::vector<uint64_t> idx;
        py::array_t<uint64_t> retip(inf.size + 1);
        auto retipptr = (uint64_t *)retip.request().ptr;
        retipptr[0] = 0;
        for(Py_ssize_t i = 0; i < inf.size; ++i) {
            int64_t ind;
            if(idx_is_4byte) ind = ((uint32_t *)inf.ptr)[i];
            else             ind = ((uint64_t *)inf.ptr)[i];
            size_t start, stop;
            if(ip_is_64) {
                start = ((uint64_t *)x.indptrp_)[ind];
                stop = ((uint64_t *)x.indptrp_)[ind + 1];
            } else {
                start = ((uint32_t *)x.indptrp_)[ind];
                stop = ((uint32_t *)x.indptrp_)[ind + 1];
            }
            size_t lastnnz = retipptr[i];
            retipptr[i + 1] = stop - start + retipptr[i];
            vals.resize(retipptr[i + 1]);
            idx.resize(retipptr[i + 1]);
            if(std::tolower(x.indices_t_[0]) == 'i') {
                std::copy((uint32_t *)x.indicesp_ + start, (uint32_t *)x.indicesp_ + stop, idx.data() + lastnnz);
            } else {
                std::copy((uint64_t *)x.indicesp_ + start, (uint64_t *)x.indicesp_ + stop, idx.data() + lastnnz);
            }
            switch(std::tolower(x.data_t_[0])) {
                case 'f': std::copy((float *)x.datap_ + start, (float *)x.datap_ + stop, vals.data() + lastnnz); break;
                case 'd': std::copy((double *)x.datap_ + start, (double *)x.datap_ + stop, vals.data() + lastnnz); break;
                case 'L': case 'l': std::copy((uint64_t *)x.datap_ + start, (uint64_t *)x.datap_ + stop, vals.data() + lastnnz); break;
                case 'I': case 'i': std::copy((uint32_t *)x.datap_ + start, (uint32_t *)x.datap_ + stop, vals.data() + lastnnz); break;
            }
        }
        py::array_t<double> retv(vals.size());
        std::copy(vals.begin(), vals.end(), (double *)retv.request().ptr);
        py::array_t<uint64_t> reti(idx.size());
        std::copy(idx.begin(), idx.end(), (uint64_t *)reti.request().ptr);
        return py::make_tuple(retv, reti, retip, py::make_tuple(py::int_(inf.size), py::int_(x.columns())));
    });

     m.def("kmeanspp", [](const PyCSparseMatrix &smw, const SumOpts &so, py::object weights) {
        return run_kmpp_noso(smw, py::int_(int(so.dis)), py::int_(int(so.k)),  so.gamma, so.seed, so.kmc2_rounds, std::max(int(so.extra_sample_tries) - 1, 0),
                       so.lspp, so.use_exponential_skips, weights);
    },
    "Computes a selecion of points from the matrix pointed to by smw, returning indexes for selected centers, along with assignments and costs for each point.",
       py::arg("smw"),
       py::arg("opts"),
       py::arg("weights") = py::none()
    );
    m.def("kmeanspp",  run_kmpp_noso,
       "Computes a selecion of points from the matrix pointed to by smw, returning indexes for selected centers, along with assignments and costs for each point."
       "\nSet nkmc to -1 to perform streaming kmeans++ (kmc2 over the full dataset), which parallelizes better but may yield a lower-quality result.\n",
       py::arg("smw"), py::arg("msr"), py::arg("k"), py::arg("betaprior") = 0., py::arg("seed") = 0, py::arg("nkmc") = 0, py::arg("ntimes") = 1,
       py::arg("lspp") = 0, py::arg("use_exponential_skips") = false,
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
#endif
}
