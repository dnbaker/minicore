#include "pyfgc.h"
#include "smw.h"
#include "pyhelpers.h"
using blaze::unaligned;
using blaze::unpadded;


template<typename FT, typename WFT>
py::dict cpp_pycluster(const blz::SM<FT> &mat, unsigned int k, double beta,
               dist::DissimilarityMeasure measure,
               WFT *weights=static_cast<WFT *>(nullptr),
               double eps=1e-10,
               int ntimes=3,
               uint64_t seed = 13,
               unsigned lspprounds=0,
               size_t kmcrounds=1000,
               size_t kmeansmaxiter=1000);

template<typename WFT>
py::dict pycluster(const SparseMatrixWrapper &smw, int k, double beta,
               dist::DissimilarityMeasure measure,
               WFT *weights,
               double eps=1e-10,
               int ntimes=3,
               uint64_t seed = 13,
               unsigned lspprounds=0,
               size_t kmcrounds=1000,
               size_t kmeansmaxiter=1000)
{
    assert(k >= 1);
    assert(beta > 0.);
    if(smw.is_float()) return cpp_pycluster(smw.getfloat(), k, beta, measure, weights, eps, ntimes, seed, lspprounds, kmcrounds, kmeansmaxiter);
    return cpp_pycluster(smw.getdouble(), k, beta, measure, weights, eps, ntimes, seed, lspprounds, kmcrounds, kmeansmaxiter);
}


template<typename FT, typename WFT, typename CtrT, typename AsnT=blz::DV<uint32_t>, typename CostsT=blz::DV<double>>
py::dict cpp_pycluster_from_centers(const blz::SM<FT> &mat, unsigned int k, double beta,
               dist::DissimilarityMeasure measure,
               std::vector<CtrT> &ctrs,
               AsnT &asn, CostsT &costs,
               WFT *weights,
               double eps,
               size_t kmeansmaxiter)
{
    std::fprintf(stderr, "[%s]\n", __PRETTY_FUNCTION__);
    blz::DV<FT> prior{FT(beta)};
    auto [initcost, finalcost, numiter] = perform_hard_clustering(mat, measure, prior, ctrs, asn, costs, weights, eps, kmeansmaxiter);
    auto pyctrs = centers2pylist(ctrs);
    auto pycosts = vec2fnp<decltype(costs), float> (costs);
    auto pyasn = vec2fnp<decltype(asn), uint32_t>(asn);
    return py::dict("initcost"_a = initcost, "finalcost"_a = finalcost, "numiter"_a = numiter,
                    "centers"_a = pyctrs, "costs"_a = pycosts, "asn"_a=pyasn);
}

template<typename WFT, typename CtrT, typename AsnT=blz::DV<uint32_t>, typename CostsT=blz::DV<double>>
py::dict cpp_pycluster_from_centers(const SparseMatrixWrapper &mat, unsigned int k, double beta,
               dist::DissimilarityMeasure measure,
               std::vector<CtrT> &ctrs,
               AsnT &asn, CostsT &costs,
               WFT *weights,
               double eps,
               size_t kmeansmaxiter)
{
    py::dict ret;
    if(mat.is_float())
        ret = cpp_pycluster_from_centers(mat.getfloat(), k, beta, measure, ctrs, asn, costs, weights, eps, kmeansmaxiter);
    else
        ret = cpp_pycluster_from_centers(mat.getdouble(), k, beta, measure, ctrs, asn, costs, weights, eps, kmeansmaxiter);
    return ret;
}

template<typename FT, typename WFT>
py::dict cpp_pycluster(const blz::SM<FT> &mat, unsigned int k, double beta,
               dist::DissimilarityMeasure measure,
               WFT *weights,
               double eps,
               int ntimes,
               uint64_t seed ,
               unsigned lspprounds,
               size_t kmcrounds,
               size_t kmeansmaxiter)
{
    std::fprintf(stderr, "[%s] beginning cpp_pycluster\n", __PRETTY_FUNCTION__);
    blz::DV<FT> prior{FT(beta)};
    const FT psum = beta * mat.columns();
    auto cmp = [measure, psum,&prior](const auto &x, const auto &y) {
        // Note that this has been transposed
        return cmp::msr_with_prior(measure, y, x, prior, psum, blz::sum(y), blz::sum(x));
    };
    if(measure != dist::L1 && measure != dist::L2 && measure != dist::BHATTACHARYYA_METRIC) {
        std::fprintf(stderr, "D2 sampling may not provide a bicriteria approximation alone. TODO: use more expensive metric clustering for better objective functions.\n");
    }
    wy::WyRand<uint32_t> rng(seed);
    std::fprintf(stderr, "About to try to get initial centers\n");
    auto initial_sol = repeatedly_get_initial_centers(mat, rng, k, kmcrounds, ntimes, cmp);
    std::fprintf(stderr, "Got initial centers\n");
    auto &[idx, asn, costs] = initial_sol;
    std::vector<blz::CompressedVector<FT, blz::rowVector>> centers(k);
    for(unsigned i = 0; i < k; ++i)
        centers[i] = row(mat, idx[i]);
    return cpp_pycluster_from_centers(mat, k, beta, measure, centers, asn, costs, weights, eps, kmeansmaxiter);
}

template<typename VecT>
void set_centers(VecT *vec, const py::buffer_info &bi) {
    auto &v = *vec;
    switch(bi.format.front()) {
        case 'f':
        for(Py_ssize_t i = 0; i < bi.shape[0]; ++i) {
            blaze::CustomVector<float, unaligned, unpadded> cv((float *)bi.ptr + i * bi.shape[1], bi.shape[1]);
            v.emplace_back(trans(cv));
        }
        break;
        case 'd':
        for(Py_ssize_t i = 0; i < bi.shape[0]; ++i) {
            blaze::CustomVector<double, unaligned, unpadded> cv((double *)bi.ptr + i * bi.shape[1], bi.shape[1]);
            v.emplace_back(trans(cv));
        }
        break;
        default: throw std::invalid_argument(std::string("Invalid format string: ") + bi.format);
    }
}

void init_clustering(py::module &m) {
    auto func1 = [](const SparseMatrixWrapper &smw, int k, double beta,
                        py::object msr, py::object weights, double eps,
                        int ntimes, uint64_t seed, int lspprounds, int kmcrounds, uint64_t kmeansmaxiter)
    {
        if(beta < 0) beta = 1. / smw.columns();
        dist::DissimilarityMeasure measure;
        if(py::isinstance<py::int_>(msr)) {
            measure = static_cast<dist::DissimilarityMeasure>(
                py::cast<Py_ssize_t>(msr));
        } else measure = dist::str2msr(py::cast<std::string>(msr));
        std::fprintf(stderr, "Beginning pycluster (v1)\n");
        if(weights.is_none()) {
            if(smw.is_float())
                return pycluster(smw, k, beta, measure, (blz::DV<float> *)nullptr, eps, ntimes, seed, lspprounds, kmcrounds, kmeansmaxiter);
            else
                return pycluster(smw, k, beta, measure, (blz::DV<double> *)nullptr, eps, ntimes, seed, lspprounds, kmcrounds, kmeansmaxiter);
        }
        auto weightinfo = py::cast<py::array>(weights).request();
        if(weightinfo.itemsize == 4)  {
            blz::CustomVector<float, unaligned, unpadded> cv((float *)weightinfo.ptr, smw.rows());
            return pycluster(smw, k, beta, measure, &cv, eps, ntimes, seed, lspprounds, kmcrounds, kmeansmaxiter);
        } else if(weightinfo.itemsize == 8) {
            blz::CustomVector<double, unaligned, unpadded> cv((double *)weightinfo.ptr, smw.rows());
            return pycluster(smw, k, beta, measure, &cv, eps, ntimes, seed, lspprounds, kmcrounds, kmeansmaxiter);
        }
        throw std::invalid_argument("Weights were not float, double, or None.");
    };
    m.def("cluster", func1,
    py::arg("smw"), py::arg("k"), py::arg("betaprior") = -1., py::arg("msr") = 5, py::arg("weights") = py::none(),
    py::arg("ntimes") = 2,
    py::arg("eps") = 1e-10, py::arg("seed") = 13,
    py::arg("lspprounds") = 1, py::arg("kmcrounds") = 10000, py::arg("kmeansmaxiter") = 1000);

    m.def("cluster", [](const SparseMatrixWrapper &smw, py::object centers, double beta,
                        py::object msr, py::object weights, double eps,
                        uint64_t kmeansmaxiter, size_t kmcrounds)
    -> py::object
    {
        if(py::isinstance<py::int_>(centers)) {
            return func1(smw, centers.cast<int>(), beta, msr, weights, eps, kmeansmaxiter, kmcrounds);
        }
        if(beta < 0) beta = 1. / smw.columns();
        blz::DV<double> prior{double(beta)};
        dist::DissimilarityMeasure measure;
        if(py::isinstance<py::int_>(msr)) {
            measure = static_cast<dist::DissimilarityMeasure>(
                py::cast<Py_ssize_t>(msr));
        } else measure = dist::str2msr(py::cast<std::string>(msr));
        std::fprintf(stderr, "Beginning pycluster (v2)\n");
        std::unique_ptr<std::vector<blz::CompressedVector<double, blz::rowVector>>>dptr;
        std::unique_ptr<std::vector<blz::CompressedVector<float, blz::rowVector>>> fptr;
        const bool isf = smw.is_float();
        if(isf) fptr.reset(new std::vector<blz::CompressedVector<float, blz::rowVector>>);
        else    dptr.reset(new std::vector<blz::CompressedVector<double, blz::rowVector>>);
        std::fprintf(stderr, "Created %cptr\n", isf ? 'f': 'd');
        if(py::isinstance<py::array>(centers)) {
            auto cbuf = py::cast<py::array>(centers).request();
            if(dptr) set_centers(dptr.get(), cbuf);
            else     set_centers(fptr.get(), cbuf);
        } else if(py::isinstance<py::list>(centers)) {
            for(auto item: centers) {
                auto ca = py::cast<py::array>(item);
                auto bi = ca.request();
                const auto fmt = bi.format[0];
                auto emp = [&](auto &x) {
                    if(fptr) {
                        fptr->emplace_back();
                        fptr->back() = trans(x);
                    } else {
                        dptr->emplace_back();
                        dptr->back() = trans(x);
                    }
                };
                if(fmt == 'd') {
                    auto cv = blaze::CustomVector<double, unaligned, unpadded>((double *)bi.ptr, bi.size);
                    emp(cv);
                } else {
                    auto cv = blaze::CustomVector<float, unaligned, unpadded>((float *)bi.ptr, bi.size);
                    emp(cv);
                }
            }
        } else throw std::invalid_argument("Centers must be a 2d numpy array or a list of numpy arrays");
        const unsigned k = fptr ? fptr->size(): dptr->size();
        blz::DV<uint32_t> asn(smw.rows());
        const auto psum = beta * smw.columns();
        blz::DV<double> centersums = blaze::generate(k, [&dptr,&fptr](auto x) {
            double ret;
            if(dptr) ret = blz::sum((*dptr)[x]);
            else     ret = blz::sum((*fptr)[x]);
            return ret;
        });
        blz::DV<double> costs = blaze::generate(smw.rows(), [&](size_t idx) {
            double bestcost;
            uint32_t bestind;
            smw.perform([&](auto &mat) {
                auto r = row(mat, idx);
                const double rsum = blz::sum(r);
                bestind = 0;
                if(fptr) {
                    auto &fp = *fptr;
                    auto c = cmp::msr_with_prior(measure, r, fp[0], prior, psum, rsum, centersums[0]);
                    for(unsigned j = 1; j < k; ++j) {
                        auto nextc = cmp::msr_with_prior(measure, r, fp[j], prior, psum, rsum, centersums[j]);
                        if(nextc < c)
                            c = nextc, bestind = j;
                    }
                    bestcost = c;
                } else {
                    auto &dp = *dptr;
                    auto c = cmp::msr_with_prior(measure, r, dp[0], prior, psum, rsum, centersums[0]);
                    for(unsigned j = 1; j < k; ++j) {
                        auto nextc = cmp::msr_with_prior(measure, r, dp[j], prior, psum, rsum, centersums[j]);
                        if(nextc < c)
                            c = nextc, bestind = j;
                    }
                    bestcost = c;
                }
            });
            asn[idx] = bestind;
            return bestcost;
        });
        if(weights.is_none()) {
            if(fptr) return cpp_pycluster_from_centers(smw, k, beta, measure, *fptr, asn, costs, (blz::DV<float> *)nullptr, eps, kmeansmaxiter); 
            else     return cpp_pycluster_from_centers(smw, k, beta, measure, *dptr, asn, costs, (blz::DV<double> *)nullptr, eps, kmeansmaxiter); 
        }
        auto weightinfo = py::cast<py::array>(weights).request();
        if(weightinfo.itemsize == 4)  {
            blz::CustomVector<float, unaligned, unpadded> cv((float *)weightinfo.ptr, smw.rows());
            if(fptr) return cpp_pycluster_from_centers(smw, k, beta, measure, *fptr, asn, costs, &cv, eps, kmeansmaxiter);
            else     return cpp_pycluster_from_centers(smw, k, beta, measure, *dptr, asn, costs, &cv, eps, kmeansmaxiter);
        } else if(weightinfo.itemsize == 8) {
            blz::CustomVector<double, unaligned, unpadded> cv((double *)weightinfo.ptr, smw.rows());
            if(fptr) return cpp_pycluster_from_centers(smw, k, beta, measure, *fptr, asn, costs, &cv, eps, kmeansmaxiter);
            else     return cpp_pycluster_from_centers(smw, k, beta, measure, *dptr, asn, costs, &cv, eps, kmeansmaxiter);
        }
        throw std::invalid_argument("Weights were not float, double, or None.");
        return py::none();
    },
    py::arg("smw"),
    py::arg("centers"),
    py::arg("betaprior") = -1.,
    py::arg("msr") = 5,
    py::arg("weights") = py::none(),
    py::arg("eps") = 1e-10,
    py::arg("maxiter") = 1000,
    py::arg("kmcrounds") = 10000);

} // init_clustering
