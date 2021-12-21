#include "pyfgc.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "aesctr/wy.h"
#include "minicore/minicore.h"
#include <mutex>


template<typename T>
shared::flat_hash_map<T, std::vector<size_t>> get_map(const T *items, const py::ssize_t nelem) {
    using MT = shared::flat_hash_map<T,std::vector<size_t>>;
    int nt = 1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        nt = omp_get_num_threads();
    }
#endif
    MT map;
    if(nt == 1) {
        for(py::ssize_t i = 0; i < nelem; ++i) {
            map[items[i]].push_back(i);
        }
        return map;
    }
    std::vector<MT> maps(nt);
    OMP_PFOR
    for(py::ssize_t i = 0; i < nelem; ++i) {
        const int my_id = OMP_ELSE(omp_get_thread_num(), 0);
        auto &mymap = maps[my_id];
        const T v = items[i];
        auto it = mymap.find(v);
        if(it == mymap.end()) {
            it = mymap.emplace(v, std::vector<size_t>{}).first;
        }
        it->second.push_back(i);
    }
    for(const auto &x: maps) {
        for(const auto &pair: x) {
            auto mit = map.find(pair.first);
            if(mit == map.end()) {
                map.emplace(pair.first, pair.second);
                //mit[pair.first] = pair.second;
            } else {
                mit->second.insert(mit->second.end(), pair.second.begin(), pair.second.end());
            }
        }
    }
    return map;
}

#if 0
T get_argmaxcount(const char dt, const void *items, const py::ssize_t nelem) {
    shared::flat_hash_map<T, std::pair<std::vector<size_t>, std::unique_ptr<std::mutex>>> map = get_map(items, nelem);
    for(py::ssize_t i = 0; i < nelem; ++i) {
        const T v = items[i];
        auto it = map.find(v);
        if(it == map.end()) {
            std::pair<std::vector<size_t>, std::unique_ptr<std::mutex>> value({}, new std::mutex);
            it = map.emplace(v, std::move(value)).first;
        }
        auto &rhs = it->second;
        std::lock_guard<std::mutex> lock(*rhs.second);
        rhs.first.push_back(i);
    }
}
#endif



template<typename T>
py::dict get_counthist(const T *items, const py::ssize_t nelem) {
    //std::fprintf(stderr, "Starting [%s]\n", __PRETTY_FUNCTION__);
    //std::fprintf(stderr, "Read all %zu items\n", nelem);
    const auto map = get_map(items, nelem);
    //std::fprintf(stderr, "Filled map of siz %zu\n", map.size());
    const std::string dt = size2dtype(nelem);
    const auto retdt = py::dtype(dt);
    //std::fprintf(stderr, "dt: %s\n", dt.data());
    py::dict ret;
    for(const auto &pair: map) {
        const py::ssize_t nmatches = pair.second.size();
        py::array cpy(retdt, std::vector<py::ssize_t>{nmatches});
        const auto bi = cpy.request();
        //std::fprintf(stderr, "Data = %s format\n", bi.format.data());
        const size_t *startp = pair.second.data(), *endp = startp + nmatches;
        assert(endp - startp == pair.second.size());
        //std::fprintf(stderr, "retdt %c. %zu matches\n", dt.front(), nmatches);
        switch(dt.front()) {
            case 'l': case 'L': std::copy(startp, endp, (uint64_t *)bi.ptr); break;
            case 'i': case 'I': std::copy(startp, endp, (uint32_t *)bi.ptr); break;
            case 'h': case 'H': std::copy(startp, endp, (uint16_t *)bi.ptr); break;
            case 'b': case 'B': std::copy(startp, endp, (uint8_t *)bi.ptr); assert(bi.itemsize == 1); break;
            default: std::fprintf(stderr, "datatype %s/%c found no match.\n", dt.data(), dt.front()); std::exit(1);
        }
        //std::fprintf(stderr, "Copied over for key = %Lg\n", static_cast<long double>(pair.first));
        ret[py::cast(pair.first)] = cpy;
    }
    return ret;
}

template<typename T>
T get_argmax(const shared::flat_hash_map<T, std::vector<size_t>> &input) {
    size_t maxc = 0;
    T asn = std::numeric_limits<T>::max();
    for(const auto &item: input) {
        if(const size_t ct = item.second.size(); ct > maxc) {
            maxc = ct;
            asn = item.first;
        }
    }
    return asn;
}

template<typename T>
T get_argmax(const py::dict input) {
    size_t maxc = 0;
    T asn = std::numeric_limits<T>::max();
    for(const auto item: input) {
        if(item.second.cast<T>() > maxc) {
            maxc = item.second;
            asn = item.first;
        }
    }
    return asn;
}

py::object get_counthist1d(const py::buffer_info &bi) {
    if(bi.ndim != 1) throw std::runtime_error("Expected 1-d Numpy Array");
    py::object ret = py::none();
    switch(standardize_dtype(bi.format).front()) {
        case 'f': ret = get_counthist((float *)bi.ptr, bi.size); break;
        case 'd': ret = get_counthist((double *)bi.ptr, bi.size); break;
        case 'B': case 'b': ret = get_counthist((uint8_t *)bi.ptr, bi.size); break;
        case 'H': case 'h': ret = get_counthist((uint16_t *)bi.ptr, bi.size); break;
        case 'I': case 'i': ret = get_counthist((uint32_t *)bi.ptr, bi.size); break;
        case 'L': case 'l': ret = get_counthist((uint64_t *)bi.ptr, bi.size); break;
        default: throw std::invalid_argument(std::string("Unexpected dtype: ") + bi.format);
    }
    return ret;
}

py::object get_counthist_wrapper(py::array rhs) {
    py::list ret;
    py::buffer_info bi = rhs.request();
    if(bi.ndim == 1)
        return get_counthist1d(bi);
    if(bi.ndim != 2) throw std::runtime_error("Expected 2-d Numpy Array");
    const py::ssize_t nc = bi.shape[1], nr = bi.shape[0];
    for(py::ssize_t i = 0; i < nr; ++i) {
        py::object next = py::none();
        const uint8_t *startp = (uint8_t *)bi.ptr + bi.strides[0];
        switch(standardize_dtype(bi.format).front()) {
            case 'f': next = get_counthist((float *)startp, nc); break;
            case 'd': next = get_counthist((double *)startp, nc); break;
            case 'B': case 'b': next = get_counthist((uint8_t *)startp, nc); break;
            case 'H': case 'h': next = get_counthist((uint16_t *)startp, nc); break;
            case 'I': case 'i': next = get_counthist((uint32_t *)startp, nc); break;
            case 'L': case 'l': next = get_counthist((uint64_t *)startp, nc); break;
            default: throw std::invalid_argument(std::string("Unexpected dtype: ") + bi.format);
        }
        ret.append(next);
    }
    return ret;
}


template<typename T>
py::array_t<T> get_argmaxcount(const py::buffer_info &bi) {
    if(bi.ndim != 2) throw std::invalid_argument("Expected 2-d array for get_argcountmax");
    py::array_t<T> ret({bi.shape[0]});
    py::buffer_info rbi = ret.request();
    const size_t rowlen = bi.strides[0] / sizeof(T);
    const py::ssize_t nc = bi.shape[1], nr = bi.shape[0];
    OMP_PFOR
    for(py::ssize_t i = 0; i < nr; ++i) {
        const T *srcptr = (const T *)bi.ptr + (i * rowlen);
        ((T *)rbi.ptr)[i] = get_argmax(get_map(srcptr, nc));
    }
    return ret;
}

py::object get_argmaxcount_base(const py::buffer_info &bi) {
    switch(standardize_dtype(bi.format).front()) {
        case 'f': return get_argmaxcount<float>(bi);
        case 'd': return get_argmaxcount<double>(bi);
        case 'b': return get_argmaxcount<int8_t>(bi);
        case 'B': return get_argmaxcount<uint8_t>(bi);
        case 'h': return get_argmaxcount<int16_t>(bi);
        case 'H': return get_argmaxcount<uint16_t>(bi);
        case 'i': return get_argmaxcount<int32_t>(bi);
        case 'I': return get_argmaxcount<uint32_t>(bi);
        case 'l': return get_argmaxcount<int64_t>(bi);
        case 'L': return get_argmaxcount<uint64_t>(bi);
        default: ;
    }
    return py::none();
}

#if 0
py::object countmax(py::array rhs) {
    py::buffer_info bi = rhs.request();
    if(bi.ndim == 1) {
        return py::int_(get_argmax<size_t>(get_counthist1d(bi)));
    }
    const auto dtc = standardize_dtype(bi.format).front();
    py::array ret = py::array(py::dtype(bi.format), std::vector<py::ssize_t>{bi.shape[0]});
    py::buffer_info retbi = ret.request();
    const size_t offset = bi.strides[0] / bi.itemsize;
    OMP_PFOR
    for(size_t i = 0; i < bi.shape[0]; ++i) {
        const uint8_t *ptr = (const uint8_t *)bi.ptr + offset * i;
        auto map = get_argmax(dtc, ptr, bi.shape[1]);
        ret[i] = get_argmax<
    }
}

py::array get_asn(py::array rhs) {
    py::object list ret;
    py::list ret;
    py::buffer_info bi = rhs.request();
    const auto dtc = standardize_dtype(bi.format).front();
    if(bi.ndim == 1)
        return get_counthist(bi);
    py::array ret(dtc
    if(bi.ndim != 2) throw std::runtime_error("Expected 2-d Numpy Array");
    for(py::ssize_t i = 0; i < bi.shape[0]; ++i) {
        py::object next = py::none();
        switch(dtc) {
            case 'B': case 'b': next = get_counthist((uint8_t *)bi.ptr, bi.size); break;
            case 'H': case 'h': next = get_counthist((uint16_t *)bi.ptr, bi.size); break;
            case 'I': case 'i': next = get_counthist((int32_t *)bi.ptr, bi.size); break;
            case 'L': case 'l': next = get_counthist((uint64_t *)bi.ptr, bi.size); break;
            default: throw std::invalid_argument(std::string("Unexpected dtype: ") + bi.format);
        }
        ret.append(next);
    }
    return ret;
}
#endif

void init_counthist(py::module &m) {
    m.def("get_counthist", get_counthist_wrapper);
    m.def("get_argmaxcount", get_argmaxcount_base);
}
