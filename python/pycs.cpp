#include "pyfgc.h"

void init_coreset(py::module &m) {
    py::class_<CSType>(m, "CoresetSampler")
    .def(py::init<>())
    .def("make_sampler", [](
        CSType &cs, size_t ncenters, py::array costs, INA assignments, py::object weights, uint64_t seed, int sens_)
    {
        std::fprintf(stderr, "Making sampler. %zu centers, costs has %zd items, sens = %d\n", ncenters, costs.shape(0), sens_);
        const auto sens(static_cast<minicore::coresets::SensitivityMethod>(sens_));
        py::buffer_info buf1 = costs.request();
        const uint32_t *asnp = (const uint32_t *)assignments.request().ptr;
        if(buf1.ndim != 1) throw std::runtime_error("buffer must have one dimension (reshape if necessary)");
        float *wp = nullptr;
        if(!weights.is_none()) {
            auto winf = py::cast<py::array_t<float>>(weights).request();
            wp = (float *)winf.ptr;
        }
        switch(buf1.format[0]) {
            case 'f': cs.make_sampler(costs.shape(0), ncenters, (float *)buf1.ptr, asnp, wp, seed, sens); break;
            case 'd': cs.make_sampler(costs.shape(0), ncenters, (double *)buf1.ptr, asnp, wp, seed, sens); break;
            default: throw std::invalid_argument("Costs is not double or float");
        }
    },
    "Generates a coreset sampler given a set of costs, assignments, and, optionally, weights. This can be used to generate an index coreset",
    py::arg("ncenters"), py::arg("costs"), py::arg("assignments"),
    py::arg("weights") = py::none(), py::arg("seed") = 13, py::arg("sens")=0
    ).def("get_probs", [](CSType &cs) {
        py::array_t<float> ret(cs.np_);
        std::copy(cs.probs_.get(), cs.probs_.get() + cs.np_, (float *)ret.request().ptr);
        return ret;
    }, "Create a numpy array of sampling probabilities")
    .def("sample", [](CSType &cs, Py_ssize_t size, Py_ssize_t seed) {
        if(cs.sampler_ == nullptr) throw std::invalid_argument("Can't sample without created sampler. Call make_ampler");
        if(seed == 0) seed = std::rand();
        std::fprintf(stderr, "Gathering sample\n");
        auto ret = cs.sample(size);
        std::fprintf(stderr, "Gathered sample\n");
        for(size_t i = 0; i < size; ++i) {
            std::fprintf(stderr, "weight: %g. index: %zu\n", ret.weights_[i], size_t(ret.indices_[i]));
        }
        py::array_t<float> rf(size);
        py::array_t<uint64_t> ri(size);
        std::fprintf(stderr, "Copying results back\n");
        std::copy(ret.weights_.begin(), ret.weights_.end(), (float *)rf.request().ptr);
        std::copy(ret.indices_.begin(), ret.indices_.end(), (uint64_t *)ri.request().ptr);
        return py::make_tuple(rf, ri);
    }, py::arg("size"), py::arg("seed") = 0);
}
