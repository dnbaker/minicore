#ifndef PYSCLUSTER_HEADER_H__
#define PYSCLUSTER_HEADER_H__
#include "pycluster.h"
#include "pybind11/embed.h"

namespace clust = minicore::clustering;

using blz::rowMajor;
using blz::rowVector;
using blz::unchecked;
using blz::rowwise;
using minicore::util::sum;
using blz::sum;
using blz::unpadded;
using blz::unaligned;
using blz::padded;
using blz::aligned;

template<typename Matrix, typename CtrT, typename AsnT=blz::DV<uint32_t>, typename CostsT=blz::DV<double>>
py::dict cpp_scluster(const Matrix &mat, int, double beta,
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
    blz::DV<FT> rsums = sum<rowwise>(mat);
    const auto csums = blz::evaluate(blz::generate(ctrs.size(), [&](auto idx) {return blz::sum(ctrs[idx]);}));
    const double psum = beta * mat.columns();
    costs = blaze::generate(costs.rows(), costs.columns(), [&](auto r, auto c) {
        return cmp::msr_with_prior<float>(measure, row(mat, r, unchecked), ctrs[c], prior, psum, rsums[r], csums[c]);
    });
    asn = blaze::softmax<blaze::rowwise>(costs);
    OMP_PFOR
    for(size_t i = 0; i < asn.rows(); ++i) {
        auto r = row(asn, i, unchecked);
        clust::correct_softmax(row(costs, i, unchecked), r);
    }
    blz::DV<double> cw;
    std::unique_ptr<blz::CustomVector<double, unaligned, unpadded, rowVector>> wview;
    if(weights && wdtype > 0) {
        if(wdtype != 'd') {
            cw.resize(costs.rows());
            wview.reset(new blz::CustomVector<double, unaligned, unpadded, rowVector>(cw.data(), cw.size()));
            switch(wdtype) {
                case 'f': cw = blz::make_cv((float *)weights, costs.rows()); break;
                case 'I': case 'i': cw = blz::make_cv((uint32_t *)weights, costs.rows()); break;
                case 'L': case 'l': cw = blz::make_cv((uint64_t *)weights, costs.rows()); break;
                case 'H': case 'h': cw = blz::make_cv((uint16_t *)weights, costs.rows()); break;
                case 'B': case 'b': cw = blz::make_cv((uint8_t *)weights, costs.rows()); break;
                default: throw std::invalid_argument("Required: float, double, or uint{8,16,32,64} weights");
            }
        } else wview.reset(new blz::CustomVector<double, unaligned, unpadded, rowVector>((double *)weights, costs.rows()));
    }
    // Only one version of perform_soft_clustering compiled (for double weights)
    // This takes extra memory/time to copy the weights, but halves or thirds compile-time.
    clusterret = minicore::clustering::perform_soft_clustering(mat, measure, prior, ctrs, costs, asn, temp, kmeansmaxiter, mbsize, mbn, wview.get());
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
    std::fprintf(stderr, "temp=%g, beta=%g\n", temp, beta);
    assert(beta > 0.);
    py::dict retdict;
    py::object asns = py::none(), costs = py::none();
    std::vector<blz::CompressedVector<float, blz::rowVector>> dvecs = obj2dvec(centers);
    const int k = dvecs.size();
    std::vector<Py_ssize_t> shape{Py_ssize_t(smw.rows()), k};
    assert(k >= 1);
    if(!savepref.empty()) {
        std::fprintf(stderr, "Using savepref to mmap cost matrices diretly: %s\n", savepref.data());
        std::string cpath = savepref + ".costs." + (use_float ? ".f32": ".f64") + ".npy";
        std::string apath = savepref + ".asns." + (use_float ? ".f32": ".f64") + ".npy";
        auto mmfn = py::module::import("numpy").attr("memmap");
        auto dt = py::dtype(use_float ? "f": "d");
        costs = mmfn(py::str(cpath), shape, dt);
        asns = mmfn(py::str(apath), shape, dt);
    } else {
        if(use_float) costs = py::array_t<float>({smw.rows(), smw.columns()}), asns = py::array_t<float>({smw.rows(), smw.columns()});
        else costs = py::array_t<double>({smw.rows(), smw.columns()}), asns = py::array_t<double>({smw.rows(), smw.columns()});
    }
    void *cp = py::cast<py::array>(costs).request().ptr,
         *ap = py::cast<py::array>(asns).request().ptr;
    fprintf(stderr, "cp: %p. ap: %p\n", (void *)cp, (void *)ap);
    if(use_float) {
        fprintf(stderr, "About to perform soft clustering with floats\n");
        blz::CustomMatrix<float, unaligned, unpadded, rowMajor> cm((float *)cp, smw.rows(), k);
        blz::CustomMatrix<float, unaligned, unpadded, rowMajor> am((float *)ap, smw.rows(), k);
        smw.perform([&](auto &x) {retdict = cpp_scluster(x, k, beta, measure, dvecs, cm, am, temp, kmeansmaxiter, mbsize, mbn, weights, wfmt[0]);});
    } else {
        fprintf(stderr, "About to perform soft clustering with doubles\n");
        blz::CustomMatrix<double, unaligned, unpadded, rowMajor> cm((double *)cp, smw.rows(), k);
        blz::CustomMatrix<double, unaligned, unpadded, rowMajor> am((double *)ap, smw.rows(), k);
        smw.perform([&](auto &x) {retdict = cpp_scluster(x, k, beta, measure, dvecs, cm, am, temp, kmeansmaxiter, mbsize, mbn, weights, wfmt[0]);});
    }
    retdict["costs"] = costs;
    retdict["asn"] = asns;
    return retdict;
}


#endif
