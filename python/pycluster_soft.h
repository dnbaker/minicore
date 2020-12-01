#ifndef PYSCLUSTER_HEADER_H__
#define PYSCLUSTER_HEADER_H__
#include "pycluster.h"
#include "pybind11/embed.h"

using blz::rowMajor;
using blz::rowVector;

template<typename Matrix, typename CtrT, typename AsnT=blz::DV<uint32_t>, typename CostsT=blz::DV<double>>
py::dict cpp_scluster(const Matrix &mat, int k, double beta,
               dist::DissimilarityMeasure measure,
               std::vector<CtrT> &ctrs,
               CostsT &costs,
               AsnT &asn,
               double temp,
               size_t kmeansmaxiter,
               Py_ssize_t mbsize,
               Py_ssize_t mbn,
               void *weights=static_cast<void *>(nullptr),
               char wdtype='f')
{
    using FT = double;
    blz::DV<FT> prior{FT(beta)};
    std::tuple<double, double, size_t> clusterret;
    if(wdtype == 'f' || !weights) {
        std::unique_ptr<blz::CustomVector<float, blz::unaligned, blz::unpadded, rowVector>> wview;
        if(weights) wview.reset(new blz::CustomVector<float, blz::unaligned, blz::unpadded, rowVector>((float *)weights, mat.rows()));
        clusterret = minicore::clustering::perform_soft_clustering(mat, measure, prior, ctrs, costs, asn, temp, kmeansmaxiter, mbsize, mbn, wview.get());
    } else if(wdtype == 'd') {
        auto wview = blz::CustomVector<double, unaligned, unpadded, rowVector>((double *)weights, mat.rows());
        clusterret = minicore::clustering::perform_soft_clustering(mat, measure, prior, ctrs, costs, asn, temp, kmeansmaxiter, mbsize, mbn, &wview);
    } else throw std::runtime_error("weight dtype is not float or double");
    auto &[initcost, finalcost, numiter]  = clusterret;
    auto pyctrs = centers2pylist(ctrs);
    //auto pycosts = vec2fnp<decltype(costs), float> (costs);
    //auto pyasn = vec2fnp<decltype(asn), uint32_t>(asn);
    return py::dict("initcost"_a = initcost, "finalcost"_a = finalcost, "numiter"_a = numiter,
                    "centers"_a = pyctrs);
}

template<typename Matrix>
py::dict py_scluster(const Matrix &smw,
               py::object centers,
               dist::DissimilarityMeasure measure,
               double beta,
               double temp=1.,
               size_t kmeansmaxiter=1000,
               Py_ssize_t mbsize=-1,
               Py_ssize_t mbn=10,
               std::string savepref="",
               bool use_float=true,
               void *weights = (void *)nullptr,
               std::string wfmt="f")
{
    assert(beta > 0.);
    py::dict retdict;
    py::object asns = py::none(), costs = py::none();
    std::vector<blz::CompressedVector<float, blz::rowVector>> dvecs = obj2dvec(centers);
    const int k = dvecs.size();
    std::vector<Py_ssize_t> shape{Py_ssize_t(smw.rows()), k};
    assert(k >= 1);
    if(!savepref.empty()) {
        std::string cpath = savepref + ".costs." + (use_float ? ".f32": ".f64") + ".npy";
        std::string apath = savepref + ".asns." + (use_float ? ".f32": ".f64") + ".npy";
        auto mmfn = py::module::import("numpy").attr("memmap");
        costs = mmfn(py::str(cpath), shape, py::dtype(use_float ? "f": "d"));
        asns = mmfn(py::str(cpath), shape, py::dtype(use_float ? "f": "d"));
    } else {
        if(use_float) costs = py::array_t<float>({smw.rows(), smw.columns()}), asns = py::array_t<float>({smw.rows(), smw.columns()});
        else costs = py::array_t<double>({smw.rows(), smw.columns()}), asns = py::array_t<double>({smw.rows(), smw.columns()});
    }
    void *cp = py::cast<py::array>(costs).request().ptr,
         *ap = py::cast<py::array>(asns).request().ptr;
    if(use_float) {
        blz::CustomMatrix<float, unaligned, unpadded, rowMajor> cm((float *)cp, smw.rows(), k);
        blz::CustomMatrix<float, unaligned, unpadded, rowMajor> am((float *)ap, smw.rows(), k);
        smw.perform([&](auto &x) {retdict = cpp_scluster(x, k, beta, measure, dvecs, cm, am, temp, kmeansmaxiter, mbsize, mbn, weights, wfmt[0]);});
    } else {
        blz::CustomMatrix<double, unaligned, unpadded, rowMajor> cm((double *)cp, smw.rows(), k);
        blz::CustomMatrix<double, unaligned, unpadded, rowMajor> am((double *)ap, smw.rows(), k);
        smw.perform([&](auto &x) {retdict = cpp_scluster(x, k, beta, measure, dvecs, cm, am, temp, kmeansmaxiter, mbsize, mbn, weights, wfmt[0]);});
    }
    retdict["costs"] = costs;
    retdict["asn"] = asns;
    return retdict;
}


#endif
