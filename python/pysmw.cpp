#include "smw.h"
#include "pyfgc.h"
#include <sstream>
#include <map>

using smw_t = SparseMatrixWrapper;
dist::DissimilarityMeasure assure_dm(py::object obj) {
    dist::DissimilarityMeasure ret;
    if(py::isinstance<py::str>(obj)) {
        auto s = py::cast<std::string>(obj);
        ret = dist::str2msr(s);
    } else if(py::isinstance<py::int_>(obj)) {
        ret = static_cast<dist::DissimilarityMeasure>(obj.cast<Py_ssize_t>());
    } else {
        throw std::invalid_argument("assure_dm received object containing neither a string or an integer.");
    }
    if(!dist::is_valid_measure(ret)) throw std::invalid_argument(std::to_string(ret) + " is not a valid measure");
    return ret;
}
py::tuple py_kmeanspp(const SparseMatrixWrapper &smw, py::object msr, int k, double gamma_beta, uint64_t seed, unsigned nkmc, unsigned ntimes,
                 int lspp,
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
            case 'i': case 'u': case 'f': case 'd': kind = info.format.front(); break;
            default:throw std::invalid_argument(std::string("Invalid array format: ") + info.format + ". Expected 'd', 'f', 'i', or 'u'.\n");
        }
        wptr = info.ptr;
    }
    std::fprintf(stderr, "ki: %d\n", k);
    wy::WyRand<uint64_t> rng(seed);
    const auto psum = gamma_beta * smw.columns();
    const blz::StaticVector<double, 1> prior({gamma_beta});
    auto cmp = [measure=mmsr, psum,&prior](const auto &x, const auto &y) {
        // Note that this has been transposed
        return cmp::msr_with_prior(measure, y, x, prior, psum, blz::sum(y), blz::sum(x));
    };
    py::array_t<uint32_t> ret(k);
    int retasnbits;
    if(ki <= 256) {
        retasnbits = 8;
        std::fprintf(stderr, "uint8 labels\n");
    } else if(ki <= 63356) {
        retasnbits = 16;
        std::fprintf(stderr, "uint16 labels\n");
    } else if(ki <= 0xFFFFFFFF) {
        retasnbits = 32;
        std::fprintf(stderr, "uint32 labels\n");
    } else {
        retasnbits = 64;
        std::fprintf(stderr, "uint64 labels. Are you crazy?\n");
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
            repeatedly_get_initial_centers(x, rng, k, nkmc, ntimes, lspp, cmp)
            : kind == 'f' ? repeatedly_get_initial_centers(x, rng, k, nkmc, ntimes, lspp, cmp, (const float *)wptr)
            : kind == 'd' ? repeatedly_get_initial_centers(x, rng, k, nkmc, ntimes, lspp, cmp, (const double *)wptr)
            : kind == 'u' ? repeatedly_get_initial_centers(x, rng, k, nkmc, ntimes, lspp, cmp, (const unsigned *)wptr)
            : repeatedly_get_initial_centers(x, rng, k, nkmc, ntimes, lspp, cmp, (const int *)wptr);
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

py::tuple py_kmeanspp_so(const SparseMatrixWrapper &smw, const SumOpts &sm, py::object weights) {
    return py_kmeanspp(smw, py::int_((int)sm.dis), sm.k, sm.gamma, sm.seed, sm.kmc2_rounds, std::max(sm.extra_sample_tries - 1, 0u),
                       sm.lspp, weights);
}

void init_smw(py::module &m) {
    py::class_<SparseMatrixWrapper>(m, "SparseMatrixWrapper")
    .def(py::init<py::object, py::object, py::object>(), py::arg("sparray"), py::arg("skip_empty")=false, py::arg("use_float")=true)
    .def(py::init<std::string, bool>(), py::arg("path"), py::arg("use_float") = true)
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
        char buf[1024];
        return std::string(buf, std::sprintf(buf, "Matrix of %zu/%zu elements of %s, %zu nonzeros", wrap.rows(), wrap.columns(), wrap.is_float() ? "float32": "double", wrap.nnz()));
    }).def("rows", [](SparseMatrixWrapper &wrap) {return wrap.rows();})
    .def("columns", [](SparseMatrixWrapper &wrap) {return wrap.columns();})
    .def("nonzeros", [](SparseMatrixWrapper &wrap) {return wrap.nnz();})
    .def("rowsel", [](SparseMatrixWrapper &smw, py::array idx) {
        auto info = idx.request();
        switch(info.format[0]) {
            case 'd': case 'f': throw std::invalid_argument("Unexpected type");
        }
        py::object ret;
        if(smw.is_float()) {
            py::array_t<float> arr(std::vector<size_t>{size_t(info.size), smw.columns()});
            auto ari = arr.request();
            auto mat = blaze::CustomMatrix<float, blaze::unaligned, blaze::unpadded> ((float *)ari.ptr, info.size, smw.columns());
            switch(info.itemsize) {
                case 8: {
                    mat = rows(smw.getfloat(), (uint64_t *)info.ptr, info.size); break;
                }
                case 4: {
                    mat = rows(smw.getfloat(), (uint32_t *)info.ptr, info.size); break;
                }
                default: throw std::invalid_argument("rows must be integral and of 4 or 8 bytes");
            }
            ret = arr;
        } else {
            py::array_t<double> arr(std::vector<size_t>{size_t(info.size), smw.columns()});
            auto ari = arr.request();
            auto mat = blaze::CustomMatrix<double, blaze::unaligned, blaze::unpadded>((double *)ari.ptr, info.size, smw.columns());
            switch(info.itemsize) {
                case 8: {
                    mat = rows(smw.getfloat(), (uint64_t *)info.ptr, info.size); break;
                }
                case 4: {
                    mat = rows(smw.getfloat(), (uint32_t *)info.ptr, info.size); break;
                }
                default: throw std::invalid_argument("rows must be integral and of 4 or 8 bytes");
            }
            ret = arr;
        }
        return ret;
    }, "Select rows in numpy array idx from matrix smw, returning as dense numpy arrays", py::arg("idx"))
    .def("tofile", [](SparseMatrixWrapper &lhs, std::string path) {
        lhs.tofile(path);
    }, py::arg("path"))
    .def("sum", [](SparseMatrixWrapper &wrap, int byrow, bool usefloat) -> py::object
    {
        switch(byrow) {case -1: case 0: case 1: break; default: throw std::invalid_argument("byrow must be -1 (total sum), 0 (by column) or by row (1)");}
        if(byrow == -1) {
            double ret;
            wrap.perform([&ret](const auto &x) {ret = blaze::sum(x);});
            return py::float_(ret);
        }
        py::array ret;
        Py_ssize_t nelem = byrow ? wrap.rows(): wrap.columns();
        if(usefloat) ret = py::array_t<float>(nelem);
                else ret = py::array_t<double>(nelem);
        auto bi = ret.request();
        auto ptr = bi.ptr;
        if(bi.size != nelem) {
            char buf[256];
            auto n = std::sprintf(buf, "bi size: %u. nelem: %u\n", int(bi.size), int(nelem));
            throw std::invalid_argument(std::string(buf, buf + n));
        }
        if(usefloat) {
            blaze::CustomVector<float, blz::unaligned, blz::unpadded> cv((float *)ptr, nelem);
            wrap.perform([&](const auto &x) {
                if(byrow) cv = blz::sum<blz::rowwise>(x);
                else      cv = trans(blz::sum<blz::columnwise>(x));
            });
        } else {
            blaze::CustomVector<double, blz::unaligned, blz::unpadded> cv((double *)ptr, nelem);
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
    .def(py::init<std::string, Py_ssize_t, double, std::string, double, Py_ssize_t, bool, size_t>(), py::arg("measure"), py::arg("k") = 10, py::arg("betaprior") = 0., py::arg("sm") = "BFL", py::arg("outlier_fraction")=0., py::arg("max_rounds") = 100, py::arg("kmc2n") = 0,
        py::arg("soft") = false, "Construct a SumOpts object using a string key for the measure name and a string key for the coreest construction format.")
    .def(py::init<int, Py_ssize_t, double, std::string, double, Py_ssize_t, bool, size_t>(), py::arg("measure") = 0, py::arg("k") = 10, py::arg("betaprior") = 0., py::arg("sm") = "BFL", py::arg("outlier_fraction")=0., py::arg("max_rounds") = 100, py::arg("kmc2n") = 0,
        py::arg("soft") = false, "Construct a SumOpts object using a integer key for the measure name and a string key for the coreest construction format.")
    .def(py::init<std::string, Py_ssize_t, double, int, double, Py_ssize_t, bool, size_t>(), py::arg("measure") = "L1", py::arg("k") = 10, py::arg("betaprior") = 0., py::arg("sm") = static_cast<int>(minocore::coresets::BFL), py::arg("outlier_fraction")=0., py::arg("max_rounds") = 100, py::arg("kmc2n") = 0,
        py::arg("soft") = false, "Construct a SumOpts object using a string key for the measure name and an integer key for the coreest construction format.")
    .def(py::init<int, Py_ssize_t, double, int, double, Py_ssize_t, bool, size_t>(), py::arg("measure") = 0, py::arg("k") = 10, py::arg("betaprior") = 0., py::arg("sm") = static_cast<int>(minocore::coresets::BFL), py::arg("outlier_fraction")=0., py::arg("max_rounds") = 100, py::arg("kmc2n") = 0,
        py::arg("soft") = false, "Construct a SumOpts object using a integer key for the measure name and an integer key for the coreest construction format.")
    .def("__str__", &SumOpts::to_string)
    .def("__repr__", [](const SumOpts &x) {
        std::string ret = x.to_string();
        char buf[32];
        std::sprintf(buf, "%p", (void *)&x);
        ret += std::string(". Address: ") + buf;
        return ret;
    })
    .def_readwrite("kmc2n", &SumOpts::kmc2_rounds)
    .def_readwrite("lspp", &SumOpts::lspp)
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
    m.def("kmeanspp", py_kmeanspp,
    "Computes a selecion of points from the matrix pointed to by smw, returning indexes for selected centers, along with assignments and costs for each point."
       "\nSet nkmc to -1 to perform streaming kmeans++ (kmc2 over the full dataset), which parallelizes better but may yield a lower-quality result.\n",
       py::arg("smw"), py::arg("msr"), py::arg("k"), py::arg("betaprior") = 0., py::arg("seed") = 0, py::arg("nkmc") = 0, py::arg("ntimes") = 1,
       py::arg("lspp") = 0,
       py::arg("weights") = py::none()
    );
    m.def("kmeanspp", [](const SparseMatrixWrapper &smw, const SumOpts &so, py::object weights) {return py_kmeanspp_so(smw, so, weights);},
    "Computes a selecion of points from the matrix pointed to by smw, returning indexes for selected centers, along with assignments and costs for each point.",
       py::arg("smw"),
       py::arg("opts"),
       py::arg("weights") = py::none()
    );
    m.def("d2_select",  [](SparseMatrixWrapper &smw, const SumOpts &so, py::object weights) {
        std::vector<uint32_t> centers, asn;
        std::vector<double> dc;
        std::vector<float> fc;
        double *wptr = nullptr;
        float *fwptr = nullptr;
        if(py::isinstance<py::array>(weights)) {
            auto inf = py::cast<py::array>(weights).request();
            switch(inf.format.front()) {
                case 'd': wptr = (double *)inf.ptr; break;
                case 'f': fwptr = (float *)inf.ptr; break;
                default: throw std::invalid_argument("Wrong format weights");
            }
        }
        if(wptr) {
            if(smw.is_float())
                std::tie(centers, asn, fc) = minocore::m2d2(smw.getfloat(), so, wptr);
            else 
                std::tie(centers, asn, dc) = minocore::m2d2(smw.getdouble(), so, wptr);
        } else if(fwptr) {
            if(smw.is_float())
                std::tie(centers, asn, fc) = minocore::m2d2(smw.getfloat(), so, fwptr);
            else 
                std::tie(centers, asn, dc) = minocore::m2d2(smw.getdouble(), so, fwptr);
        } else {
            if(smw.is_float())
                std::tie(centers, asn, fc) = minocore::m2d2(smw.getfloat(), so);
            else 
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
       py::arg("smw"), py::arg("sumopts"), py::arg("weights") = py::none());
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

} // init_smw
