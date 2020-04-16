#ifndef FGC_PACKED_H__
#define FGC_PACKED_H__
#include <initializer_list>
#include <utility>
#include <tuple>


#ifndef FGC_PACKED
#  if __GNUC__ || __clang__
#    define FGC_PACKED __attribute__((__packed__))
#  else
#    define FGC_PACKED
#  endif
#endif

namespace packed {

template<typename T1, typename T2>
struct pair {
    T1 first;
    T2 second;
    pair(T1 f, T2 sec): first(f), second(sec) {}
    pair(): first(), second() {}
    template<class _U1, class _U2>
         pair(_U1&& __x, _U2&& __y)
     : first(std::forward<_U1>(__x)),
       second(std::forward<_U2>(__y)) { }
    pair(pair &&o)     = default;
    pair(const pair &o) = default;
    template<class _U1, class _U2>
        pair(pair<_U1, _U2>&& __p): first(std::move(__p.first)),
                                                  second(std::move(__p.second)) { }
    template<typename V>
    pair(std::initializer_list<V> l): first(std::move(l[0])), second(std::move(l[1])) {}
    pair &operator=(const pair &o) {
        first = o.first; second = o.second; return *this;
    }
    bool operator<(const pair &o) const {return std::tie(first, second) < std::tie(o.first, o.second);}
    bool operator>(const pair &o) const {return std::tie(first, second) > std::tie(o.first, o.second);}
    bool operator<=(const pair &o) const {return std::tie(first, second) <= std::tie(o.first, o.second);}
    bool operator>=(const pair &o) const {return std::tie(first, second) >= std::tie(o.first, o.second);}
    bool operator==(const pair &o) const {return std::tie(first, second) == std::tie(o.first, o.second);}
    bool operator!=(const pair &o) const {return std::tie(first, second) != std::tie(o.first, o.second);}
} FGC_PACKED;

template<typename T1, typename T2, typename T3>
struct triple {
    T1 first;
    T2 second;
    T3 third;
    triple(T1 f, T2 sec, T3 thi): first(f), second(sec), third(thi) {}
    triple(): first(), second(), third() {}
    template<class _U1, class _U2, class _U3>
         triple(_U1&& __x, _U2&& __y, _U3&& __z)
     : first(std::forward<_U1>(__x)),
       second(std::forward<_U2>(__y)),
       third(std::forward<_U3>(__z)) { }
    triple(triple &&o)     = default;
    triple(const triple &o) = default;
    template<class _U1, class _U2, class _U3>
        triple(triple<_U1, _U2, _U3>&& __p): first(std::move(__p.first)),
                                             second(std::move(__p.second)),
                                             third(std::move(__p.third)) { }
    template<typename V>
    triple(std::initializer_list<V> l): first(std::move(l[0])), second(std::move(l[1])), third(std::move(l[2])) {}
    triple &operator=(const triple &o) {
        first = o.first; second = o.second; third = o.third; return *this;
    }
    bool operator<(const triple &o) const {return std::tie(first, second, third) < std::tie(o.first, o.second, o.third);}
    bool operator>(const triple &o) const {return std::tie(first, second, third) > std::tie(o.first, o.second, o.third);}
    bool operator<=(const triple &o) const {return std::tie(first, second, third) <= std::tie(o.first, o.second, o.third);}
    bool operator>=(const triple &o) const {return std::tie(first, second, third) >= std::tie(o.first, o.second, o.third);}
    bool operator==(const triple &o) const {return std::tie(first, second, third) == std::tie(o.first, o.second, o.third);}
    bool operator!=(const triple &o) const {return std::tie(first, second, third) != std::tie(o.first, o.second, o.third);}
} FGC_PACKED;

} // namespace packed


#endif /* FGC_PACKED_H__ */
