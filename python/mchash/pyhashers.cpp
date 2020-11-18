#include "smw.h"
#include "pycsparse.h"

template<typename FT, template<typename> class Hasher>
auto project_array(const Hasher<FT> &hasher, const py::object &obj, bool round=false) {
    auto bi = py::cast<py::array>(obj).request();
    auto nd = bi.ndim;
    py::object ret = py::none();
    FT *ptr = (FT *)bi.ptr;
    const ssize_t nh = hasher.nh();
    if(bi.itemsize != sizeof(FT)) throw std::invalid_argument("Sanity check: itemsize and type size are different");
    if((std::is_same_v<FT, double> && bi.format.front() != 'd') || (std::is_same_v<FT, float> && bi.format.front() != 'f'))
        throw std::invalid_argument(std::string("Type of array ") + bi.format + " does not match hasher with items of size " + std::to_string(bi.itemsize));
    if(nd == 1) {
        py::array_t<FT> rv(nh);
        auto rvb = rv.request();
        blaze::CustomVector<FT, blaze::unaligned, blaze::unpadded> cv((FT *)rvb.ptr, nh);
        blaze::CustomVector<FT, blaze::unaligned, blaze::unpadded> dv(ptr, bi.size);
        if(round)
            cv = hasher.hash(dv);
        else
            cv = hasher.project(dv);
        ret = rv;
    } else if(nd == 2) {
        const ssize_t nr = bi.shape[0];
        py::array_t<FT> rv(
            py::buffer_info(nullptr, sizeof(FT), py::format_descriptor<FT>::format(),
                            2,
                            std::vector<Py_ssize_t>{nr, nh}, // shape
                            // strides
                            std::vector<Py_ssize_t>{Py_ssize_t(sizeof(FT) * nh),
                                                    Py_ssize_t(sizeof(FT))}
            )
        );
        auto rvb = rv.request();
        blaze::CustomMatrix<FT, blaze::unaligned, blaze::unpadded> cm((FT *)rvb.ptr, nr, nh);
        blaze::CustomMatrix<FT, blaze::unaligned, blaze::unpadded> dm(ptr, nr, bi.shape[1]);
        if(round) cm = hasher.hash(dm);
        else      cm = hasher.project(dm);
        ret = rv;
    } else {
        throw std::invalid_argument("Wrong number of dimensions (must be 1 or 2)");
    }
    return ret;
}

template<typename FT, template<typename> class Hasher>
auto project_array(const Hasher<FT> &hasher, const PyCSparseMatrix &smw, bool round=false) -> py::object {
    const ssize_t nh = hasher.nh();
    const ssize_t nr = smw.rows();
    py::array_t<FT> rv(
        py::buffer_info(nullptr, sizeof(FT), py::format_descriptor<FT>::format(),
                        2,
                        std::vector<Py_ssize_t>{nr, nh}, // shape
                        // strides
                        std::vector<Py_ssize_t>{Py_ssize_t(sizeof(FT) * nh),
                                                Py_ssize_t(sizeof(FT))}
        )
    );
    auto rvinf = rv.request();
    const bool iptr_is_32 = std::tolower(smw.indptr_t_[0]) == 'i';
    const bool idx_is_32 = std::tolower(smw.indices_t_[0]) == 'i';
    for(ssize_t i = 0; i < smw.rows(); ++i) {
        ssize_t start, stop;
        if(iptr_is_32) {
            start = ((uint32_t *)smw.indptr_)[i];
            stop = ((uint32_t *)smw.indptr_)[i + 1];
        } else {
            start = ((uint64_t *)smw.indptr_)[i];
            stop = ((uint64_t *)smw.indptr_)[i + 1];
        }
        switch(smw.data_t_[0]) {
            case 'd':
                if(idx_is_32) hasher.project((double *)smw.datap_ + start, (uint32_t *)smw.indicesp_ + start, stop - start, ((FT *)rvinf)[i * nh]);
                else          hasher.project((double *)smw.datap_ + start, (uint64_t *)smw.indicesp_ + start, stop - start, ((FT *)rvinf)[i * nh]);
            break;
            case 'f':
                if(idx_is_32) hasher.project((float *)smw.datap_ + start, (uint32_t *)smw.indicesp_ + start, stop - start, ((FT *)rvinf)[i * nh]);
                else          hasher.project((float *)smw.datap_ + start, (uint64_t *)smw.indicesp_ + start, stop - start, ((FT *)rvinf)[i * nh]);
            break;
        }
        if(round) {
            auto view = blz::make_cv((FT *)rvinf.ptr, nr * nh);
            view = floor(view);
        }
    }
    return rv;
}

#if 0
template<typename FT, template<typename> class Hasher>
auto project_array(const Hasher<FT> &hasher, const SparseMatrixWrapper &smw, bool round=false) -> py::object {
    const ssize_t nh = hasher.nh();
    const ssize_t nr = smw.rows();
    py::array_t<FT> rv(
        py::buffer_info(nullptr, sizeof(FT), py::format_descriptor<FT>::format(),
                        2,
                        std::vector<Py_ssize_t>{nr, nh}, // shape
                        // strides
                        std::vector<Py_ssize_t>{Py_ssize_t(sizeof(FT) * nh),
                                                Py_ssize_t(sizeof(FT))}
        )
    );
    auto rvb = rv.request();
    blaze::CustomMatrix<FT, blaze::unaligned, blaze::unpadded> cm((FT *)rvb.ptr, nr, nh);
    smw.perform([&](const auto &x){if(round) cm = hasher.hash(x); else cm = hasher.project(x);});
    return rv;
}
#endif

#define SETTINGS_GETTER(type, ftype) def("settings", [](const type<ftype> &hasher) -> LSHasherSettings {return hasher.settings_;}, "Get settings struct from hasher")
#define SMW_HASH_DEC(type, ftype) def("hash", [](const type<ftype> &hasher, const PyCSparseMatrix &smw) {return project_array(hasher, smw, true);}) \
                                 .def("project", [](const type<ftype> &hasher, const PyCSparseMatrix &smw) {return project_array(hasher, smw, false);})

void init_hashers(py::module &m) {
#if 0
    py::class_<LSHasherSettings>(m, "LSHSettings")
        .def(py::init<unsigned, unsigned, unsigned>(), py::arg("dim"), py::arg("k"), py::arg("l"))
        .def_readwrite("dim_", &LSHasherSettings::dim_)
        .def_readwrite("k_", &LSHasherSettings::k_)
        .def_readwrite("l_", &LSHasherSettings::l_)
        .def("nhashes", [](const LSHasherSettings &x) -> int {return x.k_ * x.l_;})
        .def("__eq__", [](const LSHasherSettings &lh, const LSHasherSettings &rh) {return lh == rh;})
        .def("__ne__", [](const LSHasherSettings &lh, const LSHasherSettings &rh) {return lh != rh;});

    py::class_<JSDLSHasher<double>>(m, "JSDLSHasher")
        .def(py::init<unsigned, unsigned, unsigned, double, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("r") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, double, uint64_t>(), py::arg("settings"), py::arg("r") = .1, py::arg("seed") = 0)
        .def("project", [](const JSDLSHasher<double> &hasher, py::object obj) -> py::object {
            return project_array(hasher, obj);
        })
        .def("hash", [](const JSDLSHasher<double> &hasher, py::object obj) -> py::object {
            return project_array(hasher, obj, true);
        })
        .SETTINGS_GETTER(JSDLSHasher, double)
        .SMW_HASH_DEC(JSDLSHasher, double);
    py::class_<S2JSDLSHasher<double>>(m, "S2JSDLSHasher")
        .def(py::init<unsigned, unsigned, unsigned, double, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, double, uint64_t>(), py::arg("settings"), py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const S2JSDLSHasher<double> &hasher, py::object obj) {
            return project_array(hasher, obj);
        }).SETTINGS_GETTER(S2JSDLSHasher, double)
        .SMW_HASH_DEC(S2JSDLSHasher, double)
        .def("hash", [](const S2JSDLSHasher<double> &hasher, py::object obj) {return project_array(hasher, obj, true);});
    py::class_<L1LSHasher<double>>(m, "L1LSHasher")
        .def(py::init<unsigned, unsigned, unsigned, double, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, double, uint64_t>(), py::arg("settings"), py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const L1LSHasher<double> &hasher, py::object obj) {
            return project_array(hasher, obj);
        }).SETTINGS_GETTER(L1LSHasher, double)
        .SMW_HASH_DEC(L1LSHasher, double)
        .def("hash", [](const L1LSHasher<double> &hasher, py::object obj) {return project_array(hasher, obj, true);});
    py::class_<ClippedL1LSHasher<double>>(m, "ClippedL1LSHasher")
        .def(py::init<unsigned, unsigned, unsigned, double, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, double, uint64_t>(), py::arg("settings"), py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const ClippedL1LSHasher<double> &hasher, py::object obj) {
            return project_array(hasher, obj);
        }).SETTINGS_GETTER(ClippedL1LSHasher, double)
        .def("hash", [](const ClippedL1LSHasher<double> &hasher, py::object obj) {return project_array(hasher, obj, true);})
        .SMW_HASH_DEC(ClippedL1LSHasher, double);
    py::class_<L2LSHasher<double>>(m, "L2LSHasher")
        .def(py::init<unsigned, unsigned, unsigned, double, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, double, uint64_t>(), py::arg("settings"), py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const L2LSHasher<double> &hasher, py::object obj) {
            return project_array(hasher, obj);
         }).SETTINGS_GETTER(L2LSHasher, double)
        .SMW_HASH_DEC(L2LSHasher, double)
        .def("hash", [](const L2LSHasher<double> &hasher, py::object obj) {return project_array(hasher, obj, true);});
    py::class_<LpLSHasher<double>>(m, "LpLSHasher")
        .def(py::init<unsigned, unsigned, unsigned, double, double, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("p")=1.1, py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, double, double, uint64_t>(), py::arg("settings"), py::arg("p")=1.1, py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const LpLSHasher<double> &hasher, py::object obj) {return project_array(hasher, obj);})
        .SETTINGS_GETTER(L2LSHasher, double)
        .SMW_HASH_DEC(LpLSHasher, double)
        .def("hash", [](const LpLSHasher<double> &hasher, py::object obj) {return project_array(hasher, obj, true);});
#endif
// To undo this, uncomment the above and suffix the hashers below with _f to signify use of floats
// To save compilation time, we're only allowing floats
    py::class_<JSDLSHasher<float>>(m, "JSDLSHasher")
        .def(py::init<unsigned, unsigned, unsigned, float, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("r") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, float, uint64_t>(), py::arg("settings"), py::arg("r") = .1, py::arg("seed") = 0)
        .def("project", [](const JSDLSHasher<float> &hasher, py::object obj) -> py::object {
            return project_array(hasher, obj, false);
        })
        .def("hash", [](const JSDLSHasher<float> &hasher, py::object obj) -> py::object {
            return project_array(hasher, obj, true);
        })
        .SETTINGS_GETTER(JSDLSHasher, float)
        .SMW_HASH_DEC(JSDLSHasher, float);
    py::class_<S2JSDLSHasher<float>>(m, "S2JSDLSHasher")
        .def(py::init<unsigned, unsigned, unsigned, float, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, float, uint64_t>(), py::arg("settings"), py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const S2JSDLSHasher<float> &hasher, py::object obj) {
            return project_array(hasher, obj, false);
        })
        .def("hash", [](const S2JSDLSHasher<float> &hasher, py::object obj) {
            return project_array(hasher, obj, true);
        })
        .SETTINGS_GETTER(S2JSDLSHasher, float)
        .SMW_HASH_DEC(S2JSDLSHasher, float);
    py::class_<L1LSHasher<float>>(m, "L1LSHasher")
        .def(py::init<unsigned, unsigned, unsigned, float, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, float, uint64_t>(), py::arg("settings"), py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const L1LSHasher<float> &hasher, py::object obj) {
            return project_array(hasher, obj, false);
        }).SETTINGS_GETTER(L1LSHasher, float)
        .def("hash", [](const L1LSHasher<float> &hasher, py::object obj) {
            return project_array(hasher, obj, true);
        })
        .SMW_HASH_DEC(L1LSHasher, float);
    py::class_<ClippedL1LSHasher<float>>(m, "ClippedL1LSHasher")
        .def(py::init<unsigned, unsigned, unsigned, float, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, float, uint64_t>(), py::arg("settings"), py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const ClippedL1LSHasher<float> &hasher, py::object obj) {
            return project_array(hasher, obj, false);
        })
        .def("hash", [](const ClippedL1LSHasher<float> &hasher, py::object obj) {
            return project_array(hasher, obj, true);
        })
        .SETTINGS_GETTER(ClippedL1LSHasher, float)
        .SMW_HASH_DEC(ClippedL1LSHasher, float);
    py::class_<L2LSHasher<float>>(m, "L2LSHasher")
        .def(py::init<unsigned, unsigned, unsigned, float, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, float, uint64_t>(), py::arg("settings"), py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const L2LSHasher<float> &hasher, py::object obj) {
            return project_array(hasher, obj, false);
         })
        .def("hash", [](const L2LSHasher<float> &hasher, py::object obj) {
            return project_array(hasher, obj, true);
         }).SETTINGS_GETTER(L2LSHasher, float)
        .SMW_HASH_DEC(L2LSHasher, float);
    py::class_<LpLSHasher<float>>(m, "LpLSHasher")
        .def(py::init<unsigned, unsigned, unsigned, float, float, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("p")=1.1, py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, float, float, uint64_t>(), py::arg("settings"), py::arg("p")=1.1, py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const LpLSHasher<float> &hasher, py::object obj) {return project_array(hasher, obj);})
        .SETTINGS_GETTER(LpLSHasher, float)
        .SMW_HASH_DEC(LpLSHasher, float);
}

#undef SETTINGS_GETTER
#undef SMW_HASH_DEC
PYBIND11_MODULE(minilsh, m) {
    init_hashers(m);
    m.doc() = "Python bindings for FGC, which allows for calling coreset/clustering code from numpy and converting results back to numpy arrays";
}
