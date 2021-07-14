#include "pyfgc.h"

using blz::geomedian;

template<typename FT>
auto pygeomedian(py::array_t<FT, py::array::c_style | py::array::forcecast> data, py::object weights, FT eps) {
    auto dbuf = data.request();
    if(dbuf.ndim != 2) throw std::runtime_error("Expected 2-d array");
    py::ssize_t ndret = dbuf.shape[1];
    py::array_t<FT> ret(ndret);
    auto rbuf = ret.request();
    FT *retp = (FT *)rbuf.ptr;
    FT *srcp = (FT *)dbuf.ptr;
    blaze::CustomMatrix<FT, blaze::unaligned, blaze::unpadded> cm(srcp, dbuf.shape[0], dbuf.shape[1], dbuf.strides[0] / sizeof(FT));
    blaze::CustomVector<FT, blaze::unaligned, blaze::unpadded, blaze::rowVector> cv(retp, dbuf.shape[1]);
    if(weights.is_none()) {
        geomedian(cm, cv, eps);
    } else {
        auto wa = py::cast<py::array>(weights);
        auto wbi = wa.request();
         void *wptr = wbi.ptr;
        if(wbi.format.size() != 1) {
            throw std::invalid_argument(std::string("unexpected format string ") + wbi.format);
        }
        if(wbi.shape[0] != dbuf.shape[0]) throw std::invalid_argument("Mismatched shapes");
        switch(wbi.format.front()) {
            case 'f': geomedian(cm, cv, static_cast<float *>(wptr), eps); break;
            case 'd': geomedian(cm, cv, static_cast<double *>(wptr), eps); break;
            case 'i': case 'I': case 'q': case 'Q': case 'l': case 'L': case 'h': case 'H': case 'b': case 'B':
                switch(wbi.itemsize) {
                    case 1: geomedian(cm, cv, static_cast<uint8_t *>(wptr), eps); break;
                    case 2: geomedian(cm, cv, static_cast<uint16_t *>(wptr), eps); break;
                    case 4: geomedian(cm, cv, static_cast<uint32_t *>(wptr), eps); break;
                    case 8: geomedian(cm, cv, static_cast<uint64_t *>(wptr), eps); break;
                    default: throw std::invalid_argument("Not expected: anything");
                }
                break;
            default: throw std::invalid_argument(std::string("Type for weights ") + wbi.format + "is unsupported. Supported: float, double, or {u,}int{16,32,64}.");
        }
    }
    return ret;
}

void init_centroid(py::module &m) {
    m.def("geomed", [](py::array data, py::object weights, double eps) {
        auto dbi = data.request();
        if(dbi.format.size() != 1u) throw std::invalid_argument(std::string("dbi format is the wrong size: ") + dbi.format);
        if(dbi.shape.size() != 2) throw std::invalid_argument(std::string("dbi shape has the wrong number of dimensions. Expected 2, found ") + std::to_string(dbi.shape.size()));
        switch(dbi.format.front()) {
            case 'f': pygeomedian<float>(py::cast<py::array_t<float>>(data), weights, eps); break;
            case 'd': pygeomedian<double>(py::cast<py::array_t<double>>(data), weights, eps); break;
            default: throw std::invalid_argument(std::string("dbi format is not floating point.: ") + dbi.format);
        }
    }, "Compute geometric median for a 2-d array. Optional arguments:\n\n Weights: 1-d array, defaults to unweighted. Must be nonnegative.\n epsilon: parameter for termination (default: 0 / precision)\n",
    py::arg("data"), py::arg("weights") = py::none(), py::arg("eps") = 0.);
}
