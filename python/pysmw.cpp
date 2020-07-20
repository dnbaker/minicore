#include "smw.h"
#include "pyfgc.h"
#include <sstream>


void init_smw(py::module &m) {
    py::class_<SparseMatrixWrapper>(m, "SparseMatrixWrapper")
    .def(py::init<py::object, py::object, py::object>(), py::arg("sparray"), py::arg("skip_empty")=false, py::arg("use_float")=false)
    .def("is_float", [](SparseMatrixWrapper &wrap) {
        return wrap.is_float();
    })
    .def("is_double", [](SparseMatrixWrapper &wrap) {
        return wrap.is_double();
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
        char buf[1024];
        return std::string(buf, std::sprintf(buf, "Matrix of %zu/%zu elements of %s, %zu nonzeros", wrap.rows(), wrap.columns(), wrap.is_float() ? "float32": "double", wrap.nnz()));
    })
    .def("__repr__", [](SparseMatrixWrapper &wrap) {
        wrap.perform([&](auto &x) {
            std::stringstream ss; ss << x;
            return ss.str();
        });
    }).def("rows", [](SparseMatrixWrapper &wrap) {return wrap.rows();}
    ).def("columns", [](SparseMatrixWrapper &wrap) {return wrap.columns();});


    // Utilities
    m.def("valid_measures", []() {
        py::array_t<uint32_t> ret(sizeof(dist::USABLE_MEASURES) / sizeof(dist::USABLE_MEASURES[0]));
        std::transform(std::begin(dist::USABLE_MEASURES), std::end(dist::USABLE_MEASURES), (uint32_t *)ret.request().ptr, [](auto x) {return static_cast<uint32_t>(x);});
        return ret;
    });
    m.def("meas2desc", [](int x) -> std::string {
        return dist::prob2desc((dist::DissimilarityMeasure)x);
    });
    m.def("meas2str", [](int x) -> std::string {
        return dist::prob2str((dist::DissimilarityMeasure)x);
    });
    m.def("display_measures", [](){
        for(const auto _m: dist::USABLE_MEASURES) {
            std::fprintf(stderr, "%d\t%s\t%s\n", static_cast<int>(_m), prob2str(_m), prob2desc(_m));
        }
    });
    m.def("mdict", []() {
        py::dict ret;
        for(const auto d: dist::USABLE_MEASURES) {
            ret[dist::prob2str(d)] = static_cast<Py_ssize_t>(d);
        }
        return ret;
    });

    // SumOpts
    // Used for providing a pythonic interface for summary options
    py::class_<SumOpts>(m, "SumOpts")
    .def(py::init<std::string, Py_ssize_t, double, std::string, double, Py_ssize_t, bool>(), py::arg("measure"), py::arg("k") = 10, py::arg("beta") = 0., py::arg("sm") = "BFL", py::arg("outlier_fraction")=0., py::arg("max_rounds") = 100,
        py::arg("soft") = false, "Construct a SumOpts object using a string key for the measure name and a string key for the coreest construction format.")
    .def(py::init<int, Py_ssize_t, double, std::string, double, Py_ssize_t, bool>(), py::arg("measure") = 0, py::arg("k") = 10, py::arg("beta") = 0., py::arg("sm") = "BFL", py::arg("outlier_fraction")=0., py::arg("max_rounds") = 100,
        py::arg("soft") = false, "Construct a SumOpts object using a integer key for the measure name and a string key for the coreest construction format.")
    .def(py::init<std::string, Py_ssize_t, double, int, double, Py_ssize_t, bool>(), py::arg("measure") = "L1", py::arg("k") = 10, py::arg("beta") = 0., py::arg("sm") = static_cast<int>(minocore::coresets::BFL), py::arg("outlier_fraction")=0., py::arg("max_rounds") = 100,
        py::arg("soft") = false, "Construct a SumOpts object using a string key for the measure name and an integer key for the coreest construction format.")
    .def(py::init<int, Py_ssize_t, double, int, double, Py_ssize_t, bool>(), py::arg("measure") = 0, py::arg("k") = 10, py::arg("beta") = 0., py::arg("sm") = static_cast<int>(minocore::coresets::BFL), py::arg("outlier_fraction")=0., py::arg("max_rounds") = 100,
        py::arg("soft") = false, "Construct a SumOpts object using a integer key for the measure name and an integer key for the coreest construction format.");

    m.def("d2_select",  [](SparseMatrixWrapper &smw, const SumOpts &so) {
        std::vector<uint32_t> centers, asn;
        std::vector<double> dc;
        std::vector<float> fc;
        if(smw.is_float()) {
            std::tie(centers, asn, fc) = minocore::m2d2(smw.getfloat(), so);
        } else {
            std::tie(centers, asn, dc) = minocore::m2d2(smw.getdouble(), so);
        }
        py::array_t<uint32_t> ret(centers.size()), retasn(smw.rows());
        py::array_t<double> costs(smw.rows());
        auto rpi = ret.request(), api = retasn.request(), cpi = costs.request();
        std::copy(centers.begin(), centers.end(), (uint32_t *)rpi.ptr);
        if(fc.size()) std::copy(fc.begin(), fc.end(), (double *)cpi.ptr);
        else          std::copy(dc.begin(), dc.end(), (double *)cpi.ptr);
        std::copy(asn.begin(), asn.end(), (uint32_t *)api.ptr);
        return py::make_tuple(ret, retasn, costs);
    }, "Computes a selecion of points from the matrix pointed to by smw, returning indexes for selected centers, along with assignments and costs for each point.",
       py::arg("smw"), py::arg("sumopts"));
    m.def("d2_select",  [](py::array arr, const SumOpts &so) {
        auto bi = arr.request();
        if(bi.ndim != 2) throw std::invalid_argument("arr must have 2 dimensions");
        if(bi.format.size() != 1)
            throw std::invalid_argument("bi format must be basic");
        std::vector<uint32_t> centers, asn;
        std::vector<double> dc;
        std::vector<float> fc;
        switch(bi.format.front()) {
            case 'f': {
                blaze::CustomMatrix<float, blaze::unaligned, blaze::unpadded> cm((float *)bi.ptr, bi.shape[0], bi.shape[1], bi.strides[1]);
                std::tie(centers, asn, fc) = minocore::m2d2(cm, so);
            } break;
            case 'd': {
                blaze::CustomMatrix<double, blaze::unaligned, blaze::unpadded> cm((double *)bi.ptr, bi.shape[0], bi.shape[1], bi.strides[1]);
                std::tie(centers, asn, dc) = minocore::m2d2(cm, so);
            } break;
            default: throw std::invalid_argument("Not supported: non-double/float type");
        }
        py::array_t<uint32_t> ret(centers.size()), retasn(bi.shape[0]);
        py::array_t<double> costs(bi.shape[0]);
        auto rpi = ret.request(), api = retasn.request(), cpi = costs.request();
        std::copy(centers.begin(), centers.end(), (uint32_t *)rpi.ptr);
        if(fc.size()) std::copy(fc.begin(), fc.end(), (double *)cpi.ptr);
        else          std::copy(dc.begin(), dc.end(), (double *)cpi.ptr);
        std::copy(asn.begin(), asn.end(), (uint32_t *)api.ptr);
        return py::make_tuple(ret, retasn, costs);
    }, "Computes a selecion of points from the matrix pointed to by smw, returning indexes for selected centers, along with assignments and costs for each point.",
       py::arg("data"), py::arg("sumopts"));
    m.def("greedy_select",  [](SparseMatrixWrapper &smw, const SumOpts &so) {
        std::vector<uint32_t> centers;
        std::vector<double> dret;
        std::vector<float> fret;
        if(smw.is_float()) {
            std::tie(centers, fret) = minocore::m2greedysel(smw.getfloat(), so);
        } else {
            std::tie(centers, dret) = minocore::m2greedysel(smw.getdouble(), so);
        }
        py::array_t<uint32_t> ret(centers.size());
        py::array_t<double> costs(smw.rows());
        auto rpi = ret.request(), cpi = costs.request();
        std::copy(centers.begin(), centers.end(), (uint32_t *)rpi.ptr);
        if(fret.size()) std::copy(fret.begin(), fret.end(), (double *)cpi.ptr);
        else            std::copy(dret.begin(), dret.end(), (double *)cpi.ptr);
        return py::make_tuple(ret, costs);
    }, "Computes a greedy selection of points from the matrix pointed to by smw, returning indexes and a vector of costs for each point. To allow for outliers, use the outlier_fraction parameter of Sumopts.",
       py::arg("smw"), py::arg("sumopts"));
    m.def("greedy_select",  [](SparseMatrixWrapper &smw, const SumOpts &so) {
        std::vector<uint32_t> centers;
        std::vector<double> dret;
        std::vector<float> fret;
        if(smw.is_float()) {
            std::tie(centers, fret) = minocore::m2greedysel(smw.getfloat(), so);
        } else {
            std::tie(centers, dret) = minocore::m2greedysel(smw.getdouble(), so);
        }
        py::array_t<uint32_t> ret(centers.size());
        py::array_t<double> costs(smw.rows());
        auto rpi = ret.request(), cpi = costs.request();
        std::copy(centers.begin(), centers.end(), (uint32_t *)rpi.ptr);
        if(fret.size()) std::copy(fret.begin(), fret.end(), (double *)cpi.ptr);
        else            std::copy(dret.begin(), dret.end(), (double *)cpi.ptr);
        return py::make_tuple(ret, costs);
    }, "Computes a greedy selection of points from the matrix pointed to by smw, returning indexes and a vector of costs for each point. To allow for outliers, use the outlier_fraction parameter of Sumopts.",
       py::arg("smw"), py::arg("sumopts"));
    m.def("greedy_select",  [](py::array arr, const SumOpts &so) {
        std::vector<uint32_t> centers;
        std::vector<double> dret;
        std::vector<float> fret;
        auto bi = arr.request();
        if(bi.ndim != 2) throw std::invalid_argument("arr must have 2 dimensions");
        if(bi.format.size() != 1)
            throw std::invalid_argument("bi format must be basic");
        switch(bi.format.front()) {
            case 'f': {
                blaze::CustomMatrix<float, blaze::unaligned, blaze::unpadded> cm((float *)bi.ptr, bi.shape[0], bi.shape[1], bi.strides[1]);
                std::tie(centers, fret) = minocore::m2greedysel(cm, so);
            } break;
            case 'd': {
                blaze::CustomMatrix<double, blaze::unaligned, blaze::unpadded> cm((double *)bi.ptr, bi.shape[0], bi.shape[1], bi.strides[1]);
                std::tie(centers, dret) = minocore::m2greedysel(cm, so);
            } break;
            default: throw std::invalid_argument("Not supported: non-double/float type");
        }
        py::array_t<uint32_t> ret(centers.size());
        py::array_t<double> costs(bi.shape[0]);
        auto rpi = ret.request(), cpi = costs.request();
        std::copy(centers.begin(), centers.end(), (uint32_t *)rpi.ptr);
        if(fret.size()) std::copy(fret.begin(), fret.end(), (double *)cpi.ptr);
        else            std::copy(dret.begin(), dret.end(), (double *)cpi.ptr);
        return py::make_tuple(ret, costs);
    }, "Computes a greedy selection of points from the matrix pointed to by smw, returning indexes and a vector of costs for each point. To allow for outliers, use the outlier_fraction parameter of Sumopts.",
       py::arg("data"), py::arg("sumopts"));
}
