#ifndef PYCLUSTER_HEADER_H__
#define PYCLUSTER_HEADER_H__
#include "pyfgc.h"
#include "smw.h"
#include "pyhelpers.h"
using blaze::unaligned;
using blaze::unpadded;

py::object func1(const SparseMatrixWrapper &smw, py::int_ k, double beta,
                 py::object msr, py::object weights, double eps,
                 int ntimes, uint64_t seed, int lspprounds, int kmcrounds, uint64_t kmeansmaxiter);

template<typename FT, typename WFT>
py::dict cpp_pycluster(const blz::SM<FT> &mat, unsigned int k, double beta,
               dist::DissimilarityMeasure measure,
               WFT *weights=static_cast<WFT *>(nullptr),
               double eps=1e-10,
               int ntimes=2,
               uint64_t seed=13,
               unsigned lspprounds=0,
               bool use_exponential_skips=false,
               size_t kmcrounds=1000,
               size_t kmeansmaxiter=1000,
               Py_ssize_t mbsize=-1,
               Py_ssize_t ncheckins=-1,
               Py_ssize_t reseed_count=-1,
               bool with_rep=true);

template<typename WFT>
py::dict pycluster(const SparseMatrixWrapper &smw, int k, double beta,
               dist::DissimilarityMeasure measure,
               WFT *weights,
               double eps=1e-10,
               int ntimes=3,
               uint64_t seed = 13,
               unsigned lspprounds=0,
               size_t kmcrounds=1000,
               size_t kmeansmaxiter=1000,
               Py_ssize_t mbsize=-1,
               Py_ssize_t ncheckins=-1,
               Py_ssize_t reseed_count=-1,
               bool with_rep=true)
{
    assert(k >= 1);
    assert(beta > 0.);
    if(smw.is_float()) return cpp_pycluster(smw.getfloat(), k, beta, measure, weights, eps, ntimes, seed, lspprounds, kmcrounds, kmeansmaxiter, mbsize, ncheckins, reseed_count, with_rep);
    return cpp_pycluster(smw.getdouble(), k, beta, measure, weights, eps, ntimes, seed, lspprounds, kmcrounds, kmeansmaxiter, mbsize, ncheckins, reseed_count, with_rep);
}
#endif
