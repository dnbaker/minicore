#undef BLAZE_RANDOM_NUMBER_GENERATOR
#define BLAZE_RANDOM_NUMBER_GENERATOR std::mt19937_64
#include "blaze/Math.h"
#include <functional>
#include <numeric>
#include <cassert>
#include "pdqsort.h"
#include "alias_sampler/alias_sampler.h"
#include "aesctr/wy.h"
#include <chrono>

inline auto t() {
    return std::chrono::high_resolution_clock::now();
}
template<typename T>
auto td(T x, T y) {
    return std::abs(std::ptrdiff_t((x - y).count()));
}

template<typename T>
auto avg(const T &x) {
    return std::accumulate(x.begin(), x.end(), 0., std::plus<>());
}
template<class Iter, class Compare=std::less<>>
BLAZE_ALWAYS_INLINE void insertion_sort(Iter begin, Iter end, Compare comp=Compare()) {
    using T = typename std::iterator_traits<Iter>::value_type;

    for (Iter cur = begin + 1; cur < end; ++cur) {
        Iter sift = cur;
        Iter sift_1 = cur - 1;

        // Compare first so we can avoid 2 moves for an element already positioned correctly.
        if (comp(*sift, *sift_1)) {
            T tmp = std::move(*sift);

            do { *sift-- = std::move(*sift_1); }
            while (sift != begin && comp(tmp, *--sift_1));

            *sift = std::move(tmp);
        }
    }
}

int main(int argc, char *argv[]) {
    size_t n = argc == 1 ? 1000000uLL: std::strtoull(argv[1], nullptr, 10);
    size_t ns = argc <= 2 ? 1000uLL: std::strtoull(argv[2], nullptr, 10);
    unsigned nt = argc <= 3 ? 3: std::atoi(argv[3]);
    std::vector<float> v(n);
    const float rmi = 1. / RAND_MAX;
    for(size_t i = 0; i < v.size() ; ++i) v[i] = double(std::rand()) / rmi;
    alias::AliasSampler<float, wy::WyRand<uint32_t, 2>> as(v.begin(), v.end());
    {
        std::vector<float> v2; std::swap(v2, v);
    }
    std::vector<int> s(ns);
    std::vector<std::ptrdiff_t> times(nt);
    for(size_t i = 0; i < ns; ++i)
        s[i] = as.sample();
    decltype(t()) start, end;
    for(unsigned tt = 0; tt < nt; ++tt) {
        assert(nt & 1);
        start = t();
        for(size_t i = 0; i < ns; ++i)
            s[i] = as.sample();
        end = t();
        times[tt++] = td(end, start);
    }
    if(times.size() < 60) insertion_sort(times.begin(), times.end());
    else                  std::sort(times.begin(), times.end());
    std::fprintf(stderr, "time: %zd\n", size_t(avg(times) / 1000.));
    std::fprintf(stderr, "time: %zd\n", times[nt/2] / 1000);
    for(unsigned tt = 0; tt < nt; ++tt) {
        assert(nt & 1);
        start = t();
        as(s.data(), s.data() + ns);
        end = t();
        times[tt++] = td(end, start);
    }
    if(times.size() < 60) insertion_sort(times.begin(), times.end());
    else                  std::sort(times.begin(), times.end());
    std::fprintf(stderr, "time: %zd\n", size_t(avg(times) / 1000.));
    std::fprintf(stderr, "time: %zd\n", times[nt/2] / 1000);
    as.sort_while_sampling = true;
    for(unsigned tt = 0; tt < nt; ++tt) {
        assert(nt & 1);
        start = t();
        as(s.data(), s.data() + ns);
        end = t();
        times[tt++] = td(end, start);
    }
    if(times.size() < 60) insertion_sort(times.begin(), times.end());
    else                  std::sort(times.begin(), times.end());
    std::fprintf(stderr, "time: %zd\n", size_t(avg(times) / 1000.));
    std::fprintf(stderr, "time: %zd\n", times[nt/2] / 1000);
}
