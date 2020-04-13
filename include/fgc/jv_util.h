#ifndef JV_UTIL_H__
#define JV_UTIL_H__
#include "fgc/packed.h"

namespace fgc {

namespace shared {

template<typename T1, typename T2>
using packed_pair = packed::pair<T1, T2>;

template<typename... Types>
using packed_triple = packed::triple<Types...>;

} // namespace shared

namespace jvutil {

template<typename FT, typename IT=uint32_t>
struct edgetup: public packed::triple<FT, IT, IT> {
    // Consists of:
    // 1. Cost of edge
    // 2. Facility index
    // 3 Distance index.
    // Can be easily accessed with these member functions:
    using super = packed::triple<FT, IT, IT>;
    template<typename...A> edgetup(A &&...args): super(std::forward<A>(args)...) {}
    auto cost() const {return this->first;}
    //auto &cost()      {return this->first;}
    auto fi()   const {return this->second;}
    //auto &fi()        {return this->second;}
    auto di()   const {return this->third;}
    //auto &di()        {return this->third;}
    auto sprintf(char *buf) const {
        return std::sprintf(buf, "%f:%u:%u", cost(), fi(), di());
    }
    auto printf(std::FILE *ofp=stderr) const {
        return std::fprintf(ofp, "%f:%u:%u", cost(), fi(), di());
    }
};

}

}

#endif /* JV_UTIL_H__ */
