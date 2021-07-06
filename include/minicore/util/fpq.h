#ifndef FPQ_H__
#define FPQ_H__
#include <queue>
#include <utility>
#include <functional>
#include <vector>
#include <cstdint>

namespace minicore { namespace util {

template<typename IT=std::uint32_t, typename FT=float, typename Container=std::vector<std::pair<FT, IT>>,
         typename Cmp=std::greater<>>
struct fpq: public std::priority_queue<std::pair<FT, IT>, Container, Cmp> {
    // Farthest-point priority queue providing access to underlying constainer with getc()
    // , a reserve function and that defaults to std::greater<> for farthest points.
    using super = std::priority_queue<std::pair<FT, IT>, Container, Cmp>;
    using value_type = std::pair<FT, IT>;

    IT size_;
    fpq(IT size=0): size_(size) {reserve(size);}
    fpq(const fpq &o) = default;
    void reserve(size_t n) {this->c.reserve(n);}
    auto &getc() {return this->c;}
    const auto &getc() const {return this->c;}
    void update(const fpq &o) {
        for(const auto &v: o.getc())
            add(v);
    }
    static constexpr bool is_vector = std::is_same_v<std::vector<std::pair<FT, IT>>, Container>;
    template<typename=std::enable_if_t<is_vector>>
    std::pair<IT, FT> operator[](size_t i) const {
        return this->c[i];
    }
    void add(const value_type &v) {
        if(this->size() < size_) this->push(v);
        else if(v > this->top()) {
            this->pop();
            this->push(v);
        }
    }
    void add(FT val, IT index) {
        if(this->size() < size_) {
            this->push(value_type(val, index));
        } else if(val > this->top().first) {
            this->pop();
            this->push(value_type(val, index));
        }
    }
};

} // util
} //minicore
#endif
