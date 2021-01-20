#include "pycluster.h"


template<typename FT>
py::object __py_cluster_from_centers_dense(py::array_t<FT, py::array::c_style | py::array::forcecast> dataset,
                    py::object centers, double beta,
                    py::object msr, py::object weights, double eps,
                    uint64_t kmeansmaxiter,
                    //size_t kmcrounds, int ntimes, int lspprounds,
                    uint64_t seed,
                    Py_ssize_t mbsize, Py_ssize_t ncheckins,
                    Py_ssize_t reseed_count, bool with_rep, bool use_cs=false)
{
    blz::DV<double> prior{double(beta)};
    const dist::DissimilarityMeasure measure = assure_dm(msr);
    auto dbi = dataset.request();
    if(dbi.ndim != 2) throw std::runtime_error("Expected 2 dimensions");
    const auto dbif = standardize_dtype(dbi.format);
    int dbifmt = -1;
    switch(dbif[0]) {
        case 'd': dbifmt = 'd'; break;
        case 'f': dbifmt = 'f'; break;
        case 'i': case 'I': dbifmt = 'I'; break;
        case 'h': case 'H': dbifmt = 'H'; break;
        break;
    }
    const size_t nr = dbi.shape[0], nc = dbi.shape[1];
    std::vector<blz::DynamicVector<FT, blz::rowVector>> dvecs = obj2dvec(centers, py::cast<py::array_t<FT, py::array::c_style | py::array::forcecast>>(dataset));

    const auto k = dvecs.size();
    blz::DV<uint32_t> asn(nr);
    if(k > 0xFFFFFFFFull) throw std::invalid_argument("k must be < 4.3 billion to fit into a uint32_t");
    const auto psum = beta * nc;
    blz::DV<double> centersums(k), rsums(nr), costs(nr);
    for(size_t i = 0; i < k; ++i) centersums[i] = blz::sum(dvecs[i]);
    for(size_t i = 0; i < nr; ++i) rsums[i] = blz::sum(blz::make_cv((FT *)dbi.ptr + x * nc, nc));
    for(size_t idx = 0; idx < nr; ++idx) {
        uint32_t bestind = 0;
        const auto rsum = rsums[idx];
        double bestcost = cmp::msr_with_prior<FT>(measure, blz::make_cv((FT *)dbi.ptr + nc * idx, nc), dvecs[0], prior, psum, rsum, centersums[0]);
        for(unsigned j = 1; j < k; ++j) {
            double nextc = cmp::msr_with_prior<FT>(measure, blz::make_cv((FT *)dbi.ptr + nc * idx, nc), dvecs[j], prior, psum, rsum, centersums[j]);
            if(nextc < bestcost) bestcost = nextc, bestind = j;
        }
        asn[idx] = bestind;
        costs[idx] = bestcost;
    }
    int wk = -1;
    blz::DV<double> bw;
    std::unique_ptr<blaze::CustomVector<double, blz::unaligned, blz::unpadded>> dcv;
    if(!weights.is_none()) {
        double *bwptr = nullptr;
        py::buffer_info wbi = py::cast<py::array>(weights).request();
        wk = standardize_dtype(wbi.format)[0];
        if(wk != 'd') {
            bw.resize(costs.size());
            bwptr = bw.data();
        }
        switch(wk) {
            case 'd': bwptr = (double *)wbi.ptr; break;
            case 'f': bw = blz::make_cv((float *)wbi.ptr, costs.size()); break;
            case 'i': case 'I': bw = blz::make_cv((uint32_t *)wbi.ptr, costs.size()); break;
            case 'l': case 'L': bw = blz::make_cv((uint64_t *)wbi.ptr, costs.size()); break;
            case 'h': case 'H': bw = blz::make_cv((uint16_t *)wbi.ptr, costs.size()); break;
            case 'B': case 'b': bw = blz::make_cv((uint8_t *)wbi.ptr, costs.size()); break;
            default: throw std::invalid_argument(std::string("unexpected array kind ") + std::to_string(wk));
        }
        dcv.reset(new blaze::CustomVector<double, blz::unaligned, blz::unpadded>(bwptr, costs.size()));
    }
    // Only compile 1 version: double weights, which can take a nullable weight container
    auto dmat = blaze::CustomMatrix<FT, blz::unaligned, blz::unpadded>((FT *)dbi.ptr, nr, nc);
    return cpp_pycluster_from_centers(dmat, k, beta, measure, dvecs, asn, costs, dcv.get(), eps, kmeansmaxiter, mbsize, ncheckins, reseed_count, with_rep, seed, use_cs);
}

void init_clustering_dense(py::module &m) {

    m.def("hcluster", [](py::array dataset, py::object centers, double beta,
                         py::object msr, py::object weights, double eps,
                         uint64_t kmeansmaxiter, uint64_t seed, Py_ssize_t mbsize, Py_ssize_t ncheckins,
                         Py_ssize_t reseed_count, bool with_rep, bool use_cs) {
                            constexpr const int pyflags = py::array::c_style | py::array::forcecast;
                            py::object ret;
                            switch(standardize_dtype(dataset.request().format)[0]) {
                                case 'f': ret = __py_cluster_from_centers_dense(py::cast<py::array_t<float, pyflags>>(dataset),
                                                                                 centers, beta, msr, weights, eps, kmeansmaxiter,
                                                                                 seed, mbsize, ncheckins, reseed_count, with_rep, use_cs);
                                break;
                                case 'd': ret = __py_cluster_from_centers_dense(py::cast<py::array_t<double, pyflags>>(dataset),
                                                                                 centers, beta, msr, weights, eps, kmeansmaxiter,
                                                                                 seed, mbsize, ncheckins, reseed_count, with_rep, use_cs);
                                break;
                                default: throw std::runtime_error("Unexpected dtype");
                            }
                            return ret;
                         },
    py::arg("dataset"),
    py::arg("centers"),
    py::arg("prior") = 0.,
    py::arg("msr") = 2,
    py::arg("weights") = py::none(),
    py::arg("eps") = 1e-10,
    py::arg("maxiter") = 100,
    py::arg("seed") = 0,
    py::arg("mbsize") = Py_ssize_t(-1),
    py::arg("ncheckins") = Py_ssize_t(-1),
    py::arg("reseed_count") = Py_ssize_t(5),
    py::arg("with_rep") = false, py::arg("use_cs") = false,
    "Clusters a SparseMatrixWrapper object using settings and the centers provided above; set prior to < 0 for it to be 1 / ncolumns(). Performs seeding, followed by EM or minibatch k-means");
} // init_clustering
