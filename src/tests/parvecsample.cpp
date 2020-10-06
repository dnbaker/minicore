#define BLAZE_USE_SLEEF 1
#include "minocore/util/macros.h"
#include "sleef.h"
#include "blaze/Math.h"
#include "aesctr/wy.h"
#include <chrono>
#include <x86intrin.h>
#include <cstdint>
using std::uint64_t;

auto gett() {return std::chrono::high_resolution_clock::now();}

int main(int argc, char **argv) {
    size_t nr = argc == 1 ? 1000: std::atoi(argv[1]);
    size_t nd = argc <= 2 ? 100: std::atoi(argv[2]);
    double rmi = 1. / RAND_MAX;
    blaze::DynamicMatrix<double> mat = blaze::generate(nr, nd, [rmi](auto, auto) {return std::rand() * rmi;});
    auto start = gett();
    blaze::DynamicVector<double> cdf(nr);
    wy::WyRand<uint64_t> rng;
    std::vector<uint32_t> ind1, ind2, ind3;
    std::cauchy_distribution<double> dist;
    //blaze::DynamicVector<double> weights = blaze::generate(nr, [&](auto x) {return std::abs(dist(rng));});
    blaze::DynamicVector<double> vals(nr);
    blaze::DynamicMatrix<double> v = blaze::generate(nd, nr, [&](auto x, auto y) {return blaze::l1Norm(row(mat, y) - row(mat, x, blaze::unchecked));});
    for(size_t i = 0; i < v.rows(); ++i) {
        std::partial_sum(v.begin(i), v.end(i), cdf.begin());
        auto ind = std::lower_bound(cdf.begin(), cdf.end(), cdf[nd - 1] * std::uniform_real_distribution<double>()(rng)) - cdf.begin();
        ind1.push_back(ind);
    }
    auto stop = gett();
    long long unsigned t1 = (stop - start).count();
    std::fprintf(stderr, "Compute + cdf time: %llu\n", t1);
    long long dgt = 0, dgt0 = 0, dgt1 = 0;
    start = gett();
    for(size_t i = 0; i < v.rows(); ++i) {
        vals = blaze::generate(nr, [](auto x) {wy::WyRand<uint64_t> rng(x); return std::uniform_real_distribution<double>()(rng);});
        //vals = log(vals) / (row(v, i, blaze::unchecked) * weights);
        vals = log(vals) / trans(row(v, i, blaze::unchecked));
        uint32_t bestind = 0;
        double mv = vals[0];
        #pragma omp parallel for
        for(size_t i = 1; i < vals.size(); ++i) {
            if(vals[i] > mv) {
                #pragma omp critical
                {
                    if(vals[i] > mv) {bestind = i, mv = vals[i];}
                }
            }
        }
        ind2.push_back(bestind);
    }
    stop = gett();
#if GTTIMERS
    std::fprintf(stderr, "Times of %lld, %lld, %lld (%g, %g, %g of max)\n", dgt, dgt0, dgt1, double(dgt) / mx, double(dgt0) / mx, double(dgt1) / mx);
#endif
    long long unsigned t2 = (stop - start).count();
    std::fprintf(stderr, "Generate + log + div + max: %llu\n", t2);
    double ratio = double(t1) / t2;
    std::fprintf(stderr, "ratio: %g\n", ratio);
    start = gett();
    auto func = [](auto x) {wy::WyRand<uint64_t> rng(x); return std::uniform_real_distribution<double>()(rng);};
    for(size_t i = 0; i < v.rows(); ++i) {
        uint32_t bestind = argmax(log(blaze::generate(v.columns(), func)) / trans(row(v, i, blaze::unchecked)));
        ind3.push_back(bestind);
    }
    stop = gett();
#if GTTIMERS
    auto mx = std::max(std::max(dgt, dgt0), dgt1);
    std::fprintf(stderr, "Times of %lld, %lld, %lld (%g, %g, %g of max)\n", dgt, dgt0, dgt1, double(dgt) / mx, double(dgt0) / mx, double(dgt1) / mx);
#endif
    long long unsigned t3 = (stop - start).count();
    std::fprintf(stderr, "argmax from flow: %llu\n", t3);
    double ratio3 = double(t1) / t3;
    std::fprintf(stderr, "ratio: %g\n", ratio3);
    
    start = gett();
    for(size_t i = 0; i < v.rows(); ++i) {
#ifdef __AVX2__
        constexpr size_t nperel = sizeof(__m256d) / sizeof(double);
        const size_t e = ((nd + nperel - 1) / nperel);
        constexpr double pdmul = 1. / (1ull<<52);
        double maxv = -std::numeric_limits<double>::max();
        size_t o;
        auto myrow = row(v, i, blaze::unchecked);
        for(o = 0; o < e; ++o) {
            __m256i v;
            for(size_t j = 0; j < nperel; ++j) {
                ((uint64_t *)&v)[j] = rng();
            }
            auto v2 = _mm256_or_si256(_mm256_slli_epi64(v, 12), _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)));
            auto v3 = _mm256_sub_pd(_mm256_castsi256_pd(v2), _mm256_set1_pd(0x0010000000000000));
            auto v4 = _mm256_mul_pd(v3, _mm256_set1_pd(pdmul));
            auto v5 = Sleef_logd4_u35(v4);
            auto ov6 = _mm256_load_pd((const double *) &myrow[i * nperel]);
            auto divv = _mm256_div_pd(v5, ov6);
            _mm256_storeu_pd((double *)&vals[o * nperel], divv);
        }
        o *= nperel;
        while(o < nd) {
            vals[o] = std::log(double(rng()) / wy::WyRand<uint64_t>::max()) / myrow[o];
            ++o;
        }
#else
        vals = log(vals) / (row(v, i, blaze::unchecked);
#endif
        uint32_t bestind = 0;
        double mv = vals[0];
        #pragma omp parallel for
        for(size_t i = 1; i < vals.size(); ++i) {
            if(vals[i] > mv) {
                #pragma omp critical
                {
                    if(vals[i] > mv) {bestind = i, mv = vals[i];}
                }
            }
        }
        ind2.push_back(bestind);
    }
    stop = gett();
    long long unsigned t4 = (stop - start).count();
    std::fprintf(stderr, "argmax from manual avx2: %llu\n", t4);
    double ratio4 = double(t1) / t4;
    std::fprintf(stderr, "ratio: %g\n", ratio4);
}
