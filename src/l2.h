#ifndef L2_DETAIL_H__
#define L2_DETAIL_H__
#include "mtx2cs.h"

template<typename FT>
std::tuple<std:vector<uint32_t>, std::vector<uint32_t>, blz::DV<FT, blz::rowVector>>
l2_sum_core(blz::SM<FT> &mat, std::string out, Opts opts) {
    std::tuple<std:vector<uint32_t>, std::vector<uint32_t>, blz::DV<FT, blz::rowVector>> ret;
    throw NotImplementedError(__PRETTY_FUNCTION__);
    return ret;
}
#endif
