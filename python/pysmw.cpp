#include "smw.h"
#include "pyfgc.h"
#include <sstream>
#include <map>

using smw_t = SparseMatrixWrapper;

void init_smw(py::module &m) {
    py::class_<SparseMatrixWrapper>(m, "SparseMatrix")
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
#if 1
        char buf[1024];
        return std::string(buf, std::sprintf(buf, "Matrix of %zu/%zu elements of %s, %zu nonzeros", wrap.rows(), wrap.columns(), wrap.is_float() ? "float32": "double", wrap.nnz()));
#else
        std::string ret;
        wrap.perform([&](auto &x) {
            std::stringstream ss; ss << x;
            ret = ss.str();
        });
        return ret;
#endif
    }).def("rows", [](SparseMatrixWrapper &wrap) {return wrap.rows();}
    ).def("columns", [](SparseMatrixWrapper &wrap) {return wrap.columns();})
    .def("sum", [](SparseMatrixWrapper &wrap, int byrow, bool usefloat) -> py::object
    {
        switch(byrow) {case -1: case 0: case 1: break; default: throw std::invalid_argument("byrow must be -1 (total sum), 0 (by column) or by row (1)");}
        if(byrow == -1) {
            double ret;
            wrap.perform([&ret](const auto &x) {ret = blaze::sum(x);});
            return py::float_(ret);
        }
        py::array ret;
        if(usefloat) ret = py::array_t<float>(byrow ? wrap.rows(): wrap.columns());
                else ret = py::array_t<double>(byrow ? wrap.rows(): wrap.columns());
        auto bi = ret.request();
        auto ptr = bi.ptr;
        if(usefloat) {
            blaze::CustomVector<float, blz::unaligned, blz::unpadded> cv((float *)ptr, bi.size);
            wrap.perform([&](const auto &x) {
                if(byrow) cv = blz::sum<blz::rowwise>(x);
                else      cv = trans(blz::sum<blz::columnwise>(x));
            });
        } else {
            blaze::CustomVector<double, blz::unaligned, blz::unpadded> cv((double *)ptr, bi.size);
            wrap.perform([&](const auto &x) {
                if(byrow) cv = blz::sum<blz::rowwise>(x);
                else      cv = trans(blz::sum<blz::columnwise>(x));
            });
        }
        return ret;
    }, py::arg("kind")=-1, py::arg("usefloat")=true);


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
        py::arg("soft") = false, "Construct a SumOpts object using a integer key for the measure name and an integer key for the coreest construction format.")
    .def("__str__", &SumOpts::to_string)
    .def("__repr__", [](const SumOpts &x) {
        std::string ret = x.to_string();
        char buf[32];
        std::sprintf(buf, "%p", (void *)&x);
        ret += std::string(". Address: ") + buf;
        return ret;
    })
    .def_readwrite("gamma", &SumOpts::gamma).def_readwrite("k", &SumOpts::k)
    .def_readwrite("search_max_rounds", &SumOpts::lloyd_max_rounds).def_readwrite("extra_sample_rounds", &SumOpts::extra_sample_tries)
    .def_readwrite("soft", &SumOpts::soft)
    .def_readwrite("outlier_fraction", &SumOpts::outlier_fraction)
    .def_readwrite("discrete_metric_search", &SumOpts::discrete_metric_search)
    .def_property("cs",
            [](SumOpts &obj) -> py::str {
                return std::string(coresets::sm2str(obj.sm));
            },
            [](SumOpts &obj, py::object item) {
                Py_ssize_t val;
                if(py::isinstance<py::str>(item)) {
                    val = minocore::coresets::str2sm(py::cast<std::string>(item));
                } else if(py::isinstance<py::int_>(item)) {
                    val = py::cast<Py_ssize_t>(item);
                } else throw std::invalid_argument("value must be str or int");
                obj.sm = (minocore::coresets::SensitivityMethod)val;
            }
        )
    .def_property("prior", [](SumOpts &obj) -> py::str {
        switch(obj.prior) {
            case dist::NONE: return "NONE";
            case dist::DIRICHLET: return "DIRICHLET";
            case dist::FEATURE_SPECIFIC_PRIOR: return "FSP";
            case dist::GAMMA_BETA: return "GAMMA";
            default: throw std::invalid_argument(std::string("Invalid prior: ") + std::to_string((int)obj.prior));
        }
    }, [](SumOpts &obj, py::object asn) -> void {
        if(asn.is_none()) {
            obj.prior = dist::NONE;
            return;
        }
        if(py::isinstance<py::str>(asn)) {
            const std::map<std::string, dist::Prior> map {
                {"NONE", dist::NONE},
                {"DIRICHLET", dist::DIRICHLET},
                {"GAMMA", dist::GAMMA_BETA},
                {"GAMMA_BETA", dist::GAMMA_BETA},
                {"FSP", dist::FEATURE_SPECIFIC_PRIOR},
                {"FEATURE_SPECIFIC_PRIOR", dist::FEATURE_SPECIFIC_PRIOR},
            };
            auto key = std::string(py::cast<py::str>(asn));
            for(auto &i: key) i = std::toupper(i);
            auto it = map.find(std::string(py::cast<py::str>(asn)));
            if(it == map.end())
                throw std::out_of_range("Prior must be NONE, FSP, FEATURE_SPECIFIC_PRIOR, GAMMA, or DIRICHLET");
            obj.prior = it->second;
        } else if(py::isinstance<py::int_>(asn)) {
            auto x = py::cast<Py_ssize_t>(asn);
            if(x > 3) throw std::out_of_range("x must be <= 3 if an integer, to represent various priors");
            obj.prior = (dist::Prior)x;
        }
    });
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
