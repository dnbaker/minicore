#ifndef MINOCORE_MERGE_FUNCTIONAL_H__
#define MINOCORE_MERGE_FUNCTIONAL_H__
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cassert>

namespace minicore {
namespace merge {
using std::size_t;

/*
 *  * We provide two abstractions:
 *     for_each_by_case, which takes n, two iterators (lhs), two more iterators (rhs), and 3 functors:
 *
 *             1. FShared, which is called on (index, xval, yval) for cases where both lh and rh have nonzero entries.
 *                     2. LHF, which is called on (index, xval) for relevant cases
 *                             2. RHF, which is called on (index, yval) for relevant cases
 *
 *                                 This version returns size_t for the number
 *
 *                                    and another, which takes an additional callable
 *                                            4. ZFunc, which is called on (index) for relevant cases.
 *                                             */

template<typename IT1, typename IT2, typename FShared, typename LHF, typename RHF>
size_t for_each_by_case(const size_t n, IT1 start1, IT1 stop1, IT2 start2, IT2 stop2, const FShared &shfunc, const LHF &lhfunc, const RHF &rhfunc) {
    size_t sharedz = 0, ci = 0, nextind = 0;
    for(;;) {
        switch(((start1 != stop1) << 1) | (start2 != stop2)) {
            case 3: /* Both are not end*/
                if(start1->index() == start2->index()) {
                    nextind = start1->index();
                    shfunc(nextind, start1->value(), start2->value());
                    ++start1; ++start2;
                } else if(start1->index() < start2->index()) {
                    nextind = start1->index();
                    lhfunc(nextind, start1->value()); ++start1;
                } else {
                    nextind = start2->index();
                    rhfunc(nextind, start2->value()); ++start2;
                }
                break;
            case 2:
                assert(start1 != stop1);
                nextind = start1->index(); lhfunc(nextind, start1->value()); ++start1;
                break;
            case 1:
                assert(start2 != stop2);
                nextind = start2->index(); rhfunc(nextind, start2->value()); ++start2;
                break;
            case 0: nextind = n; break;
            default: __builtin_unreachable();
        }
        assert(nextind <= n);
        if(nextind > ci) sharedz += nextind - ci;
        ci = nextind + 1;
        if(ci >= n) {
            break;
        }
    }
    assert(sharedz < 0xFFFFFFFFFFFF0000uLL);
    return sharedz;
}

template<typename IT1, typename IT2, typename FShared, typename LHF, typename RHF, typename ZFunc>
void for_each_by_case(const size_t n, IT1 start1, IT1 stop1, IT2 start2, IT2 stop2, const FShared &shfunc, const LHF &lhfunc, const RHF &rhfunc, const ZFunc &zfunc) {
    size_t ci = 0, nextind;
    for(;;) {
        switch(((start1 != stop1) << 1) | (start2 != stop2)) {
            case 3: /* Both are not end*/
                if(start1->index() == start2->index()) {
                    nextind = start1->index();
                    shfunc(nextind, start1->value(), start2->value());
                    ++start1; ++start2;
                } else if(start1->index() < start2->index()) {
                    nextind = start1->index();
                    lhfunc(nextind, start1->value()); ++start1;
                } else {
                    nextind = start2->index();
                    rhfunc(nextind, start2->value()); ++start2;
                }
                break;
            case 2:
                nextind = start1->index(); lhfunc(nextind, start1->value()); ++start1;
                break;
            case 1:
                nextind = start2->index(); rhfunc(nextind, start2->value()); ++start2;
                break;
            case 0: nextind = n; break;
            default: __builtin_unreachable();
        }
        while(ci < nextind) zfunc(ci++);
        if((ci = nextind + 1) >= n) break;
    }
}

template<typename IT1, typename IT2, typename FShared>
size_t for_each_if_shared(const size_t n, IT1 start1, IT1 stop1, IT2 start2, IT2 stop2, const FShared &shfunc) {
    size_t sharedz = 0, ci = 0, nextind = 0;
    do {
        const int v = (int(start1 != stop1) << 1) | (start2 != stop2);
        assert(v >= 0 || v != std::fputc(v, stderr));
        assert(v <= 3 || v != std::fputc(v, stderr));
        switch(v)
        {
        case 0: nextind = n; break;
        case 1: nextind = start2->index(); ++start2; break;
        case 2: nextind = start1->index(); ++start1; break;
        case 3:
              if(start1->index() == start2->index()) {
                  nextind = start1->index();
                  shfunc(nextind, start1->value(), start2->value());
                  ++start1; ++start2;
              } else if(start1->index() < start2->index()) {
                  nextind = start1->index(); ++start1;
              } else {
                  nextind = start2->index(); ++start2;
              }
        }
        sharedz += nextind - ci;
        ci = nextind + 1;
    } while(ci < n);
    return sharedz;
}


} // namespace merge

} // namespace minicore

#endif
