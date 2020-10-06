#ifndef SIMD_SAMPLING_H
#define SIMD_SAMPLING_H
#include <x86intrin.h>
#include "blaze/Math.h"
#include <limits>
#include "minocore/util/tsg.h"
#include "minocore/util/macros.h"
#include "aesctr/wy.h"

namespace minocore {

INLINE float horizontal_max(__m128 x) {
    __m128 max1 = _mm_shuffle_ps(x, x, _MM_SHUFFLE(0,0,3,2));
    __m128 max2 = _mm_max_ps(x, max1);
    __m128 max3 = _mm_shuffle_ps(max2, max2, _MM_SHUFFLE(0,0,0,1));
    __m128 max4 = _mm_max_ps(max2, max3);
    return _mm_cvtss_f32(max4);
}

uint64_t simd_sampling(const double *__restrict__ weights, size_t n, uint64_t seed=0)
{
    uint64_t bestind;
    wy::WyRand<uint64_t> rng(seed * seed + 13);
#ifdef __AVX512F__
    constexpr size_t nperel = sizeof(__m512d) / sizeof(double);
    const size_t e = n / nperel;
    constexpr double pdmul = 1. / (1ull<<52);
    bestind = 0;
    __m512d vmaxv = _mm512_set1_pd(-std::numeric_limits<double>::max());
    size_t o;
    OMP_PFOR
    for(o = 0; o < e; ++o) {
        thread_local tsg::ThreadSeededGen<wy::WyRand<uint64_t>> rng;
        __m512i v;
        for(size_t j = 0; j < nperel; ++j) {
            ((uint64_t *)&v)[j] = rng();
        }
        auto v2 = _mm512_or_si512(_mm512_srli_epi64(v, 12), _mm512_castpd_si512(_mm512_set1_pd(0x0010000000000000)));
        auto v3 = _mm512_sub_pd(_mm512_castsi512_pd(v2), _mm512_set1_pd(0x0010000000000000));
        auto v4 = _mm512_mul_pd(v3, _mm512_set1_pd(pdmul));
        auto v5 = Sleef_logd8_u35(v4);
        auto ov6 = _mm512_load_pd((const double *)&weights[o * nperel]);
        auto divv = _mm512_div_pd(v5, ov6);
        auto cmpmask = _mm512_cmp_pd_mask(divv, vmaxv, _CMP_GT_OQ);
        if(cmpmask) {
            auto newmaxv = _mm512_set1_pd(_mm512_reduce_max_pd(divv));
            if((cmpmask = _mm512_cmp_pd_mask(divv, newmaxv, _CMP_EQ_OQ))) {
            OMP_CRITICAL
                if(_mm512_cmp_pd_mask(divv, vmaxv, _CMP_GT_OQ)) {
                    vmaxv = newmaxv;
                    bestind = __builtin_ctz(cmpmask) + o * nperel;
                }
            }
        }
    }
    double maxv = _mm512_cvtsd_f64(vmaxv);
    for(size_t p = o * nperel; p != n; ++p) {
        std::uniform_real_distribution<double> urd;
        auto v = std::log(urd(rng)) / weights[p];
        if(v > maxv)
            bestind = p, maxv = v;
    }
#elif __AVX2__
    constexpr size_t nperel = sizeof(__m256d) / sizeof(double);
    const size_t e = (n / nperel);
    constexpr double pdmul = 1. / (1ull<<52);
    bestind = 0;
    __m256d vmaxv = _mm256_set1_pd(-std::numeric_limits<double>::max());
    size_t o = 0;
    OMP_PFOR
    for(o = 0; o < e; ++o) {
        thread_local tsg::ThreadSeededGen<wy::WyRand<uint64_t>> rng;
        __m256i v;
        for(size_t j = 0; j < nperel; ++j) {
            ((uint64_t *)&v)[j] = rng();
        }
        auto v2 = _mm256_or_si256(_mm256_srli_epi64(v, 12), _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)));
        auto v3 = _mm256_sub_pd(_mm256_castsi256_pd(v2), _mm256_set1_pd(0x0010000000000000));
        auto v4 = _mm256_mul_pd(v3, _mm256_set1_pd(pdmul));
        auto v5 = Sleef_logd4_u35(v4);
        auto ov6 = _mm256_load_pd((const double *) &weights[o * nperel]);
        auto divv = _mm256_div_pd(v5, ov6);
        auto cmp = _mm256_cmp_pd(divv, vmaxv, _CMP_GT_OQ);
        auto cmpmask = _mm256_movemask_pd(cmp);
        if(cmpmask) {
            __m256d y = _mm256_permute2f128_pd(divv, divv, 1);
            __m256d m1 = _mm256_max_pd(divv, y);
            __m256d m2 = _mm256_permute_pd(m1, 5);
            auto newmaxv = _mm256_max_pd(m1, m2);
            cmpmask = _mm256_movemask_pd(_mm256_cmp_pd(divv, newmaxv, _CMP_EQ_OQ));
            if(_mm256_movemask_pd(_mm256_cmp_pd(divv, vmaxv, _CMP_GT_OQ))) {
                OMP_CRITICAL
                if(_mm256_movemask_pd(_mm256_cmp_pd(divv, vmaxv, _CMP_GT_OQ))) {
                    vmaxv = newmaxv;
                    bestind = __builtin_ctz(cmpmask) + o * nperel;
                }
            }
        }
    }
    double maxv = _mm256_cvtsd_f64(vmaxv);
    for(size_t p = o * nperel; p != n; ++p) {
        if(!weights[p]) continue;
        std::uniform_real_distribution<double> urd;
        auto v = std::log(urd(rng)) / weights[p];
        if(v > maxv)
            bestind = p, maxv = v;
    }
#elif __SSE2__
    constexpr size_t nperel = sizeof(__m128d) / sizeof(double);
    const size_t e = n / nperel;
    constexpr double pdmul = 1. / (1ull<<52);
    double maxv = -std::numeric_limits<double>::max();
    bestind = 0;
    __m128d vmaxv = _mm_set1_pd(maxv);
    size_t o;
    OMP_PFOR
    for(o = 0; o < e; ++o) {
        thread_local tsg::ThreadSeededGen<wy::WyRand<uint64_t>> rng;
        __m128i v;
        for(size_t j = 0; j < nperel; ++j) {
            ((uint64_t *)&v)[j] = rng();
        }
        auto v2 = _mm_or_si128(_mm_srli_epi64(v, 12), _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)));
        auto v3 = _mm_sub_pd(_mm_castsi128_pd(v2), _mm_set1_pd(0x0010000000000000));
        auto v4 = _mm_mul_pd(v3, _mm_set1_pd(pdmul));
        auto v5 = Sleef_logd2_u35(v4);
        auto ov6 = _mm_load_pd((const double *) &weights[o * nperel]);
        auto divv = _mm_div_pd(v5, ov6);
        auto cmp = _mm_cmp_pd(divv, vmaxv, _CMP_GT_OQ);
        auto cmpmask = _mm_movemask_pd(cmp);
        if(cmpmask) {
            OMP_CRITICAL
            cmpmask = _mm_movemask_pd(_mm_cmp_pd(divv, vmaxv, _CMP_GT_OQ));
            if(cmpmask) {
                vmaxv = _mm_max_pd(divv, _mm_permute_pd(divv, 1));
                bestind = __builtin_ctz(_mm_movemask_pd(_mm_cmp_pd(vmaxv, divv, _CMP_EQ_OQ))) + o * nperel;
            }
        }
    }
    for(size_t p = o * nperel; p != n; ++p) {
        std::uniform_real_distribution<double> urd;
        auto v = std::log(urd(rng)) / weights[p];
        if(v > maxv)
            bestind = p, maxv = v;
    }
#else
    bestind = 0;
    double bestv = std::log(std::uniform_real_distribution<double>()(rng)) / weights[0];
    for(size_t i = 1; i < n; ++i) {
        auto v = std::log(std::uniform_real_distribution<double>()(rng)) / weights[i];
        if(v > bestv) bestv = v, bestind = i;
    }
#endif
    return bestind;
}


uint64_t simd_sampling(const float *__restrict__ weights, size_t n, uint64_t seed=0)
{
    uint64_t bestind;
    wy::WyRand<uint64_t> rng(seed * seed + 13);
#ifdef __AVX512F__
    constexpr size_t nperel = sizeof(__m512d) / sizeof(float);
    const size_t e = n / nperel;
    constexpr float psmul = 1. / (1ull<<32);
    bestind = 0;
    __m512 vmaxv = _mm512_set1_ps(-std::numeric_limits<float>::max());
    size_t o;
    OMP_PFOR
    for(o = 0; o < e; ++o) {
        thread_local tsg::ThreadSeededGen<wy::WyRand<uint64_t>> rng;
        __m512i v;
        for(size_t j = 0; j < sizeof(__m512d) / sizeof(uint64_t); ++j) {
            ((uint64_t *)&v)[j] = rng();
        }
        auto v4 = _mm512_mul_ps(_mm512_cvtepi32_ps(v), _mm512_set1_ps(psmul));
        auto v5 = Sleef_logf16_u35(v4);
        auto ov6 = _mm512_load_ps((const float *)&weights[o * nperel]);
        auto divv = _mm512_div_ps(v5, ov6);
        auto cmpmask = _mm512_cmp_ps_mask(divv, vmaxv, _CMP_GT_OQ);
        if(cmpmask) {
            auto newmaxv = _mm512_set1_ps(_mm512_reduce_max_ps(divv));
            if((cmpmask = _mm512_cmp_ps_mask(divv, newmaxv, _CMP_EQ_OQ))) {
                OMP_CRITICAL
                if(_mm512_cmp_ps_mask(divv, vmaxv, _CMP_GT_OQ)) {
                    vmaxv = newmaxv;
                    bestind = __builtin_ctz(cmpmask) + o * nperel;
                }
            }
        }
    }
    float maxv = _mm512_cvtss_f32(vmaxv);
    for(size_t p = o * nperel; p != n; ++p) {
        std::uniform_real_distribution<float> urd;
        auto v = std::log(urd(rng)) / weights[p];
        if(v > maxv)
            bestind = p, maxv = v;
    }
#elif __AVX2__
    constexpr size_t nperel = sizeof(__m256) / sizeof(float);
    const size_t e = (n / nperel);
    constexpr float psmul = 1. / (1ull<<32);
    bestind = 0;
    __m256 vmaxv = _mm256_set1_ps(-std::numeric_limits<float>::max());
    size_t o = 0;
    OMP_PFOR
    for(o = 0; o < e; ++o) {
        thread_local tsg::ThreadSeededGen<wy::WyRand<uint64_t>> rng;
        __m256i v;
        for(size_t j = 0; j < sizeof(__m256) / sizeof(uint64_t); ++j) {
            ((uint64_t *)&v)[j] = rng();
        }
        auto v2 = _mm256_mul_ps(_mm256_cvtepi32_ps(v), _mm256_set1_ps(psmul));
        auto v3 = Sleef_logf8_u35(v2);
        auto ov6 = _mm256_load_ps((const float *) &weights[o * nperel]);
        auto divv = _mm256_div_ps(v3, ov6);
        auto cmp = _mm256_cmp_ps(divv, vmaxv, _CMP_GT_OQ);
        auto cmpmask = _mm256_movemask_ps(cmp);
        if(cmpmask) {
            const __m256 permHalves = _mm256_permute2f128_ps(divv, divv, 1);
            const __m256 m0 = _mm256_max_ps(permHalves, divv);
            const __m256 perm0 = _mm256_permute_ps(m0, 0b01001110);
            const __m256 m1 = _mm256_max_ps(m0, perm0);
            const __m256 perm1 = _mm256_permute_ps(m1, 0b10110001);
            const __m256 m2 = _mm256_max_ps(perm1, m1);
            cmpmask = _mm256_movemask_ps(_mm256_cmp_ps(m2, divv, _CMP_EQ_OQ));
            OMP_CRITICAL
            if(_mm256_movemask_ps(_mm256_cmp_ps(m2, vmaxv, _CMP_GT_OQ))) {
                vmaxv = m2;
                bestind = __builtin_ctz(cmpmask) + o * nperel;
            }
        }
    }
    float maxv = _mm256_cvtss_f32(vmaxv);
    for(size_t p = o * nperel; p != n; ++p) {
        std::uniform_real_distribution<float> urd;
        auto v = std::log(urd(rng)) / weights[p];
        if(v > maxv)
            bestind = p, maxv = v;
    }
#elif __SSE2__
    constexpr size_t nperel = sizeof(__m128d) / sizeof(float);
    const size_t e = n / nperel;
    constexpr float psmul = 1. / (1ull<<32);
    float maxv = -std::numeric_limits<float>::max();
    bestind = 0;
    __m128 vmaxv = _mm_set1_ps(maxv);
    size_t o;
    OMP_PFOR
    for(o = 0; o < e; ++o) {
        thread_local tsg::ThreadSeededGen<wy::WyRand<uint64_t>> rng;
        __m128i v;
        for(size_t j = 0; j < nperel; ++j) {
            ((uint64_t *)&v)[j] = rng();
        }
        auto v3 = _mm_mul_ps(_mm_cvtepi32_ps(v), _mm_set1_ps(psmul));
        auto v5 = Sleef_logf4_u35(v3);
        auto ov6 = _mm_load_ps((const float *) &weights[o * nperel]);
        auto divv = _mm_div_ps(v5, ov6);
        auto cmp = _mm_cmp_ps(divv, vmaxv, _CMP_GT_OQ);
        auto cmpmask = _mm_movemask_ps(cmp);
        if(cmpmask) {
            OMP_CRITICAL
            if((cmpmask = _mm_movemask_ps(_mm_cmp_ps(divv, vmaxv, _CMP_GT_OQ)))) {
                vmaxv = _mm_set1_ps(horizontal_max(divv));
                bestind = __builtin_ctz(_mm_movemask_ps(_mm_cmp_ps(vmaxv, divv, _CMP_EQ_OQ))) + o * nperel;
            }
        }
    }
    for(size_t p = o * nperel; p != n; ++p) {
        std::uniform_real_distribution<float> urd;
        auto v = std::log(urd(rng)) / weights[p];
        if(v > maxv)
            bestind = p, maxv = v;
    }
#else
    bestind = 0;
    double bestv = std::log(std::uniform_real_distribution<double>()(rng)) / weights[0];
    for(size_t i = 1; i < n; ++i) {
        auto v = std::log(std::uniform_real_distribution<double>()(rng)) / weights[i];
        if(v > bestv) bestv = v, bestind = i;
    }
#endif
    return bestind;
}

template<typename WeightT>
uint64_t simd_sampling(const WeightT &weights, uint64_t seed=0) {
    return simd_sampling(weights.data(), weights.size(), seed);
}

} // namespace minocore
#endif
