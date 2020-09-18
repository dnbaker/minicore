//
// C++17 implementation of ZipIterator by Dario Pellegrini <pellegrini.dario@gmail.com>
// Originally created: October 2019
//
// Includes suggestions from https://codereview.stackexchange.com/questions/231352/c17-zip-iterator-compatible-with-stdsort
//
// Licence: Creative Commons Zero v1.0 Universal
// See LICENCE.md file accompaining this file
//

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

template <typename ...T>
class ZipRef {
protected:
  std::tuple<T*...> ptr;

  template <std::size_t I = 0>
  void copy_assign(const ZipRef& z) {
    *(std::get<I>(ptr)) = *(std::get<I>(z.ptr));
    if constexpr( I+1 < sizeof...(T) ) copy_assign<I+1>(z);
  }
  template <std::size_t I = 0>
  void val_assign(const std::tuple<T...>& t) {
    *(std::get<I>(ptr)) = std::get<I>(t);
    if constexpr( I+1 < sizeof...(T) ) val_assign<I+1>(t);
  }

public:
  ZipRef() = delete;
  ZipRef(const ZipRef& z) = default;
  ZipRef(ZipRef&& z) = default;
  ZipRef(T* const... p): ptr(p...) {}

  ZipRef& operator=(const ZipRef& z)             { copy_assign( z); return *this; }
  ZipRef& operator=(const std::tuple<T...>& val) { val_assign(val); return *this; }

  std::tuple<T...> val() const {return std::apply([](auto&&...args){ return std::tuple((*args)...); }, ptr);}
  operator std::tuple<T...>() const { return val(); }

  template <std::size_t I = 0>
  void swap_data(const ZipRef& o) const {
    std::swap(*std::get<I>(ptr), *std::get<I>(o.ptr));
    if constexpr(I+1 < sizeof...(T)) swap_data<I+1>(o);
  }

  template<std::size_t N = 0>
  decltype(auto) get() {return *std::get<N>(ptr);}
  template<std::size_t N = 0>
  decltype(auto) get() const {return *std::get<N>(ptr);}

  #define OPERATOR(OP) \
    bool operator OP(const ZipRef & o) const { return val() OP o.val(); } \
    inline friend bool operator OP(const ZipRef& r, const std::tuple<T...>& t) { return r.val() OP t; } \
    inline friend bool operator OP(const std::tuple<T...>& t, const ZipRef& r) { return t OP r.val(); }

    OPERATOR(==) OPERATOR(<=) OPERATOR(>=)
    OPERATOR(!=) OPERATOR(<)  OPERATOR(>)
  #undef OPERATOR
};

namespace std {

template<std::size_t N, typename...T>
struct tuple_element<N, ZipRef<T...>> {
    using type = decltype(std::get<N>(std::declval<ZipRef<T...>>().val()));
};

template<typename...T>
struct tuple_size<ZipRef<T...>>: public std::integral_constant<std::size_t, sizeof...(T)> {};

template<std::size_t N, typename...T>
decltype(auto) get(ZipRef<T...> &r) {
    return r.template get<N>();
}
template<std::size_t N, typename...T>
decltype(auto) get(const ZipRef<T...> &r) {
    return r.template get<N>();
}

} // namespace std

template<typename ...IT>
class ZipIter {
  std::tuple<IT...> it;

  template <std::size_t I = 0>
  bool one_is_equal(const ZipIter& rhs) const {
    if (std::get<I>(it) == std::get<I>(rhs.it)) return true;
    if constexpr(I+1 < sizeof...(IT)) return one_is_equal<I+1>(rhs);
    return false;
  }
  template <std::size_t I = 0>
  bool none_is_equal(const ZipIter& rhs) const {
    if (std::get<I>(it) == std::get<I>(rhs.it)) return false;
    if constexpr(I+1 < sizeof...(IT)) return none_is_equal<I+1>(rhs);
    return true;
  }

public:
  using iterator_category = std::common_type_t<typename std::iterator_traits<IT>::iterator_category...>;
  using difference_type   = std::common_type_t<typename std::iterator_traits<IT>::difference_type...>;
  using value_type        = std::tuple<typename std::iterator_traits<IT>::value_type ...>;
  using pointer           = std::tuple<typename std::iterator_traits<IT>::pointer ...>;
  using reference         = ZipRef<std::remove_reference_t<typename std::iterator_traits<IT>::reference>...>;

  ZipIter() = default;
  ZipIter(const ZipIter &rhs) = default;
  ZipIter(ZipIter&& rhs) = default;
  ZipIter(const IT&... rhs): it(rhs...) {}

  ZipIter& operator=(const ZipIter& rhs) = default;
  ZipIter& operator=(ZipIter&& rhs) = default;

  ZipIter& operator+=(const difference_type d) {
    std::apply([&d](auto&&...args){((std::advance(args,d)),...);}, it); return *this;
  }
  ZipIter& operator-=(const difference_type d) { return operator+=(-d); }

  reference operator* () const {return std::apply([](auto&&...args){return reference(&(*(args))...);}, it);}
  pointer   operator->() const {return std::apply([](auto&&...args){return pointer  (&(*(args))...);}, it);}
  reference operator[](difference_type rhs) const {return *(operator+(rhs));}

  ZipIter& operator++() { return operator+=( 1); }
  ZipIter& operator--() { return operator+=(-1); }
  ZipIter operator++(int) {ZipIter tmp(*this); operator++(); return tmp;}
  ZipIter operator--(int) {ZipIter tmp(*this); operator--(); return tmp;}

  difference_type operator-(const ZipIter& rhs) const {return std::get<0>(it)-std::get<0>(rhs.it);}
  ZipIter operator+(const difference_type d) const {ZipIter tmp(*this); tmp += d; return tmp;}
  ZipIter operator-(const difference_type d) const {ZipIter tmp(*this); tmp -= d; return tmp;}
  inline friend ZipIter operator+(const difference_type d, const ZipIter& z) {return z+d;}
  inline friend ZipIter operator-(const difference_type d, const ZipIter& z) {return z-d;}

  // Since operator== and operator!= are often used to terminate cycles,
  // defining them as follow prevents incrementing behind the end() of a container
  bool operator==(const ZipIter& rhs) const { return  one_is_equal(rhs); }
  bool operator!=(const ZipIter& rhs) const { return none_is_equal(rhs); }
  #define OPERATOR(OP) \
    bool operator OP(const ZipIter& rhs) const {return it OP rhs.it;}
    OPERATOR(<=) OPERATOR(>=)
    OPERATOR(<)  OPERATOR(>)
  #undef OPERATOR
};

template<typename ...Container>
class Zip {
  std::tuple<Container&...> zip;

public:
  Zip() = delete;
  Zip(const Zip& z) = default;
  Zip(Zip&& z) = default;
  Zip(Container&... z): zip(z...) {}

  #define HELPER(OP) \
    auto OP(){return std::apply([](auto&&... args){ return ZipIter((args.OP())...);}, zip);} \
    auto c##OP() const {return std::apply([](auto&&... args){ return ZipIter((args.c##OP())...);}, zip);} \
    auto OP() const {return this->c##OP();}

    HELPER( begin) HELPER( end)
    HELPER(rbegin) HELPER(rend)
  #undef HELPER
};

using std::swap;
template<typename ...T> void swap(const ZipRef<T...>& a, const ZipRef<T...>& b) { a.swap_data(b); }

#include <sstream>
template< class Ch, class Tr, class...IT, std::enable_if_t<(sizeof...(IT)>0), int> = 0>
auto& operator<<(std::basic_ostream<Ch, Tr>& os, const ZipRef<IT...>& t) {
  std::basic_stringstream<Ch, Tr> ss;
  ss << "[ ";
  std::apply([&ss](auto&&... args) {((ss << args << ", "), ...);}, t.val());
  ss.seekp(-2, ss.cur);
  ss << " ]";
  return os << ss.str();
}

