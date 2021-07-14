#include "pycluster_soft.h"

py::dict py_scluster_arr(py::array matrix,
               py::object centers,
               dist::DissimilarityMeasure measure,
               double prior,
               double temp=1.,
               size_t kmeansmaxiter=1000,
               py::ssize_t mbsize=-1,
               py::ssize_t mbn=10,
               std::string savepref="",
               void *weights = (void *)nullptr,
               std::string wfmt="f")
{
    std::unique_ptr<py::array_t<float, py::array::c_style | py::array::forcecast>> fmat;
    py::buffer_info inf = matrix.request(), fmatinf;
    std::string ifmt = standardize_dtype(inf.format);
    py::object asns = py::none(), costs = py::none();
    std::vector<blz::CompressedVector<float, blz::rowVector>> dvecs;
    py::ssize_t nr;
    if(ifmt.front() == 'd') {
        dvecs = obj2dvec(centers, blaze::CustomMatrix<double, blaze::unaligned, blaze::unpadded>((double *)inf.ptr, inf.shape[0], inf.shape[1], inf.strides[0] / sizeof(double)));
        nr = inf.shape[0];
    } else {
        if(ifmt.front() != 'f') {
            fmat.reset(new py::array_t<float, py::array::c_style | py::array::forcecast>({inf.shape[0], inf.shape[1]}));
            fmatinf = fmat->request();
        } else fmatinf = matrix.request();
        dvecs = obj2dvec(centers, blaze::CustomMatrix<float, blaze::unaligned, blaze::unpadded>((float *)fmatinf.ptr, fmatinf.shape[0], fmatinf.shape[1], fmatinf.strides[0] / sizeof(float)));
        nr = fmatinf.shape[0];
    }
    const int k = dvecs.size();
    std::vector<py::ssize_t> shape{nr, k};
    assert(k >= 1);
    if(!savepref.empty()) {
        std::fprintf(stderr, "Using savepref to mmap cost matrices diretly: %s\n", savepref.data());
        std::string cpath = savepref + ".costs.f32.npy";
        std::string apath = savepref + ".asns.f32.npy";
        auto mmfn = py::module::import("numpy").attr("memmap");
        auto dt = py::dtype("f");
        costs = mmfn(py::str(cpath), shape, dt);
        asns = mmfn(py::str(apath), shape, dt);
    } else {
        costs = py::array_t<float>(shape);
        asns = py::array_t<float>(shape);
    }
    void *cp = py::cast<py::array>(costs).request().ptr,
         *ap = py::cast<py::array>(asns).request().ptr;
    blz::CustomMatrix<float, unaligned, unpadded, rowMajor> cm((float *)cp, nr, k);
    blz::CustomMatrix<float, unaligned, unpadded, rowMajor> am((float *)ap, nr, k);
    py::dict retdict;
    if(ifmt.front() != 'd') {
        blaze::CustomMatrix<float, blaze::unaligned, blaze::unpadded> dcm((float *)fmatinf.ptr, fmatinf.shape[0], fmatinf.shape[1]);
        retdict  = cpp_scluster(dcm, k, prior, measure, dvecs, cm, am, temp, kmeansmaxiter, mbsize, mbn, weights, wfmt[0]);
    } else {
        blaze::CustomMatrix<double, blaze::unaligned, blaze::unpadded> dcm((double *)inf.ptr, inf.shape[0], inf.shape[1]);
        retdict  = cpp_scluster(dcm, k, prior, measure, dvecs, cm, am, temp, kmeansmaxiter, mbsize, mbn, weights, wfmt[0]);
    }
    retdict["costs"] = costs;
    retdict["asn"] = asns;
    return retdict;
}


void init_clustering_softdense(py::module &m) {
    m.def("scluster", [](py::array mat, py::object centers,
                    py::object measure, double prior, double temp,
                    uint64_t kmeansmaxiter, py::ssize_t mbsize, py::ssize_t mbn,
                    py::object savepref, py::object weights) -> py::object
    {
        if(prior < 0.) prior = 0.;
        void *wptr = nullptr;
        std::string wfmt = "f";
        if(!weights.is_none()) {
            auto inf = py::cast<py::array>(weights).request();
            wfmt = standardize_dtype(inf.format);
            wptr = inf.ptr;
        }
        return py_scluster_arr(mat, centers, assure_dm(measure), prior, temp, kmeansmaxiter, mbsize, mbn, static_cast<std::string>(savepref.cast<py::str>()), wptr, wfmt);
    },
    py::arg("matrix"),
    py::arg("centers"),
    py::arg("msr") = 2,
    py::arg("prior") = 0.,
    py::arg("temp") = 1.,
    py::arg("maxiter") = 1000,
    py::arg("mbsize") = py::ssize_t(-1),
    py::arg("mbn") = py::ssize_t(-1),
    py::arg("savepref") = "",
    py::arg("weights") = py::none()
    );
} // init_clustering_csr
