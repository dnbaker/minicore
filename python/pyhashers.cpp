#include "pyfgc.h"

template<typename FT, template<typename> class Hasher>
auto project_array(const Hasher<FT> &hasher, const py::object &obj) {
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
        cm = hasher.project(dm);
        ret = rv;
    } else {
        throw std::invalid_argument("Wrong number of dimensions (must be 1 or 2)");
    }
    return ret;
}

void init_hashers(py::module &m) {
    py::class_<LSHasherSettings>(m, "LSHSettings")
        .def(py::init<unsigned, unsigned, unsigned>())
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
        }).def("settings", [](const JSDLSHasher<double> &hasher) -> LSHasherSettings {return hasher.settings_;}, "Get settings struct from hasher");
    py::class_<S2JSDLSHasher<double>>(m, "S2JSDLSHasher")
        .def(py::init<unsigned, unsigned, unsigned, double, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, double, uint64_t>(), py::arg("settings"), py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const S2JSDLSHasher<double> &hasher, py::object obj) {
            return project_array(hasher, obj);
        });
    py::class_<L1LSHasher<double>>(m, "L1LSHasher")
        .def(py::init<unsigned, unsigned, unsigned, double, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, double, uint64_t>(), py::arg("settings"), py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const L1LSHasher<double> &hasher, py::object obj) {
            return project_array(hasher, obj);
        });
    py::class_<ClippedL1LSHasher<double>>(m, "ClippedL1LSHasher")
        .def(py::init<unsigned, unsigned, unsigned, double, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, double, uint64_t>(), py::arg("settings"), py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const ClippedL1LSHasher<double> &hasher, py::object obj) {
            return project_array(hasher, obj);
        });
    py::class_<L2LSHasher<double>>(m, "L2LSHasher")
        .def(py::init<unsigned, unsigned, unsigned, double, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, double, uint64_t>(), py::arg("settings"), py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const L2LSHasher<double> &hasher, py::object obj) {
            return project_array(hasher, obj);
         });
    py::class_<LpLSHasher<double>>(m, "LpLSHasher")
        .def(py::init<unsigned, unsigned, unsigned, double, double, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("p")=1.1, py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, double, double, uint64_t>(), py::arg("settings"), py::arg("p")=1.1, py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const LpLSHasher<double> &hasher, py::object obj) {return project_array(hasher, obj);});


    py::class_<JSDLSHasher<float>>(m, "JSDLSHasher_f")
        .def(py::init<unsigned, unsigned, unsigned, float, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("r") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, float, uint64_t>(), py::arg("settings"), py::arg("r") = .1, py::arg("seed") = 0)
        .def("project", [](const JSDLSHasher<float> &hasher, py::object obj) -> py::object {
            return project_array(hasher, obj);
        }).def("settings", [](const JSDLSHasher<float> &hasher) -> LSHasherSettings {return hasher.settings_;}, "Get settings struct from hasher");
    py::class_<S2JSDLSHasher<float>>(m, "S2JSDLSHasher_f")
        .def(py::init<unsigned, unsigned, unsigned, float, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, float, uint64_t>(), py::arg("settings"), py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const S2JSDLSHasher<float> &hasher, py::object obj) {
            return project_array(hasher, obj);
        });
    py::class_<L1LSHasher<float>>(m, "L1LSHasher_f")
        .def(py::init<unsigned, unsigned, unsigned, float, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, float, uint64_t>(), py::arg("settings"), py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const L1LSHasher<float> &hasher, py::object obj) {
            return project_array(hasher, obj);
        });
    py::class_<ClippedL1LSHasher<float>>(m, "ClippedL1LSHasher_f")
        .def(py::init<unsigned, unsigned, unsigned, float, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, float, uint64_t>(), py::arg("settings"), py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const ClippedL1LSHasher<float> &hasher, py::object obj) {
            return project_array(hasher, obj);
        });
    py::class_<L2LSHasher<float>>(m, "L2LSHasher_f")
        .def(py::init<unsigned, unsigned, unsigned, float, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, float, uint64_t>(), py::arg("settings"), py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const L2LSHasher<float> &hasher, py::object obj) {
            return project_array(hasher, obj);
         });
    py::class_<LpLSHasher<float>>(m, "LpLSHasher_f")
        .def(py::init<unsigned, unsigned, unsigned, float, float, uint64_t>(), py::arg("dim"), py::arg("k"), py::arg("l"), py::arg("p")=1.1, py::arg("w") = .1, py::arg("seed") = 0)
        .def(py::init<LSHasherSettings, float, float, uint64_t>(), py::arg("settings"), py::arg("p")=1.1, py::arg("w") = .1, py::arg("seed") = 0)
        .def("project", [](const LpLSHasher<float> &hasher, py::object obj) {return project_array(hasher, obj);});
}
