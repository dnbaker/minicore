#include "pycsparse.h"
#include "smw.h"
using blz::unaligned;
using blz::aligned;
using blz::padded;
using blz::unpadded;
using blz::rowMajor;

template<typename FT>
py::tuple mat2tup(py::array_t<FT, py::array::c_style | py::array::forcecast> arr, const SumOpts &so) {
    std::vector<uint64_t> centers;
    auto arri = arr.request();
    if(arri.ndim != 2) throw std::invalid_argument("Wrong number of dimensions");
    const py::ssize_t nr = arri.shape[0], nc = arri.shape[1];
    std::vector<double> dret;
    blz::CustomMatrix<FT> cm((FT *)arri.ptr, nr, nc, arri.strides[0] / sizeof(FT));
    std::tie(centers, dret) = m2greedysel(cm, so);
    auto dtstr = size2dtype(nr);
    auto ret = py::array(py::dtype(dtstr), std::vector<py::ssize_t>{nr});
    py::array_t<FT> costs(nr);
    auto rpi = ret.request(), cpi = costs.request();
    std::copy(dret.begin(), dret.end(), (FT *)cpi.ptr);
    switch(dtstr[0]) {
        case 'L': std::copy(centers.begin(), centers.end(), (uint64_t *)rpi.ptr); break;
        case 'I': std::copy(centers.begin(), centers.end(), (uint32_t *)rpi.ptr); break;
        case 'H': std::copy(centers.begin(), centers.end(), (uint16_t *)rpi.ptr); break;
        case 'B': std::copy(centers.begin(), centers.end(), (int8_t *)rpi.ptr); break;
    }
    return py::make_tuple(ret, costs);
}

void init_pydense(py::module &m) {

     m.def("kmeanspp", [](py::array_t<float, py::array::c_style | py::array::forcecast> arr, int k, py::object measure, py::object prior, py::object seed, py::object ntimes,
                          py::object lspp, py::object weights, py::object expskips, py::object local_trials) {
        auto dm = assure_dm(measure);
        auto arri = arr.request();
        if(arri.ndim != 2) throw std::invalid_argument("Wrong number of dimensions");
        blz::CustomMatrix<float, unaligned, unpadded, rowMajor> cm((float *)arri.ptr, arri.shape[0], arri.shape[1], arri.strides[0] / sizeof(float));
        std::fprintf(stderr, "Doing kmeans++ over matrix at %p with floats\n", arri.ptr);
        return py_kmeanspp_noso_dense(cm, py::int_(int(dm)), py::int_(k), prior.cast<double>(), seed.cast<py::ssize_t>(), std::max(ntimes.cast<int>() - 1, 0),
                             lspp.cast<py::ssize_t>(), expskips.cast<bool>(), local_trials.cast<py::ssize_t>(), weights);
    },
    "Computes a selecion of points from the matrix pointed to by smw, returning indexes for selected centers, along with assignments and costs for each point.",
       py::arg("matrix"),
       py::arg("k"),
       py::arg("msr") = py::int_(2),
       py::arg("prior") = 0.,
       py::arg("seed") = 0,
       py::arg("ntimes") = 0,
       py::arg("lspp") = 0,
       py::arg("weights") = py::none(),
       py::arg("expskips") = false,
       py::arg("n_local_trials") = 1
    );
     m.def("kmeanspp", [](py::array_t<double, py::array::c_style> arr, int k, py::object measure, py::object prior, py::object seed, py::object ntimes,
                          py::object lspp, py::object weights, py::object expskips, py::object local_trials) -> py::object {
        auto dm = assure_dm(measure);
        auto arri = arr.request();
        if(arri.ndim != 2) throw std::invalid_argument("Wrong number of dimensions");
        blz::CustomMatrix<double, unaligned, unpadded, rowMajor> cm((double *)arri.ptr, arri.shape[0], arri.shape[1], arri.strides[0] / sizeof(double));
        std::fprintf(stderr, "Doing kmeans++ over matrix at %p with doubles\n", arri.ptr);
        return py_kmeanspp_noso_dense(cm, py::int_(int(dm)), py::int_(k), prior.cast<double>(), seed.cast<py::ssize_t>(), std::max(ntimes.cast<int>() - 1, 0),
                             lspp.cast<py::ssize_t>(), expskips.cast<bool>(), local_trials.cast<py::ssize_t>(), weights);
    },
    "Computes a selecion of points from the matrix pointed to by smw, returning indexes for selected centers, along with assignments and costs for each point.",
       py::arg("matrix"),
       py::arg("k"),
       py::arg("msr") = py::int_(2),
       py::arg("prior") = 0.,
       py::arg("seed") = 0,
       py::arg("ntimes") = 0,
       py::arg("lspp") = 0,
       py::arg("weights") = py::none(),
       py::arg("expskips") = false,
       py::arg("n_local_trials") = 1
    );

    m.def("greedy_select", mat2tup<double>,
      "Computes a greedy selection of points from the matrix pointed to by smw, returning indexes and a vector of costs for each point. To allow for outliers, use the outlier_fraction parameter of Sumopts.",
       py::arg("matrix"), py::arg("sumopts"));
    m.def("greedy_select", mat2tup<float>,
      "Computes a greedy selection of points from the matrix pointed to by smw, returning indexes and a vector of costs for each point. To allow for outliers, use the outlier_fraction parameter of Sumopts.",
       py::arg("matrix"), py::arg("sumopts"));
    m.def("d2_select",  [](py::array mat, const SumOpts &so, py::object weights) {
        auto matinf = mat.request();
        const size_t nr = matinf.shape[0], nc = matinf.shape[1];
        std::vector<uint32_t> centers, asn;
        std::vector<double> dc;
        double *wptr = nullptr;
        blz::DV<double> tmpw;
        if(py::isinstance<py::array>(weights)) {
            auto inf = py::cast<py::array>(weights).request();
            switch(inf.format.front()) {
                case 'd': wptr = (double *)inf.ptr; break;
                case 'f': tmpw = blz::make_cv((float *)inf.ptr, nr); wptr = tmpw.data(); break;
                default: throw std::invalid_argument("Wrong format weights");
            }
        }
        auto lhs = std::tie(centers, asn, dc);
        auto mdt = standardize_dtype(matinf.format)[0];
        switch(mdt) {
            case 'f': {
                blz::CustomMatrix<float, unaligned, unpadded> cm((float *)matinf.ptr, nr, nc, matinf.strides[0] / sizeof(float));
                lhs = minicore::m2d2(cm, so, wptr);
                break;
            }
            case 'd': {
                blz::CustomMatrix<double, unaligned, unpadded> cm((double *)matinf.ptr, nr, nc, matinf.strides[0] / sizeof(double));
                lhs = minicore::m2d2(cm, so, wptr);
                break;
            }
            default: throw std::invalid_argument("Only supported: double and float types");
        }
        py::array_t<uint64_t> ret(centers.size());
        py::array_t<uint32_t> retasn(nr);
        py::array_t<double> costs(nr);
        auto rpi = ret.request(), api = retasn.request(), cpi = costs.request();
        std::copy(centers.begin(), centers.end(), (uint64_t *)rpi.ptr);
        std::copy(dc.begin(), dc.end(), (double *)cpi.ptr);
        std::copy(asn.begin(), asn.end(), (uint32_t *)api.ptr);
        return py::make_tuple(ret, retasn, costs);
    }, "Computes a selecion of points from the matrix pointed to by smw, returning indexes for selected centers, along with assignments and costs for each point.",
       py::arg("matrix"), py::arg("sumopts"), py::arg("weights") = py::none());
}
