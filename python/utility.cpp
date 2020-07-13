#include "utility.h"

dist::DissimilarityMeasure obj2m(py::str o) {
    auto s = py::cast<std::string>(py::cast<py::str>(o));
    for(auto m: dist::USABLE_MEASURES) if(!std::strcmp(dist::prob2str(m), s.data())) return m;
    throw std::invalid_argument("invalid key");
}
dist::DissimilarityMeasure obj2m(py::int_ o) {
    auto i = py::cast<Py_ssize_t>(o);
    if(!dist::is_valid_measure((dist::DissimilarityMeasure)i)) {
        std::fprintf(stderr, "Warning: measure %td is not considered valid. Downstream work may break.\n", std::ptrdiff_t(i));
    }
    return static_cast<dist::DissimilarityMeasure>(i);
}

coresets::SensitivityMethod obj2sm(py::str o) {
    auto s = py::cast<std::string>(py::cast<py::str>(o));
    for(auto m: coresets::CORESET_CONSTRUCTIONS)
        if(s == coresets::sm2str(m)) return m;
    throw std::invalid_argument("invalid key");
}

coresets::SensitivityMethod obj2sm(py::int_ o) {
    auto i = py::cast<Py_ssize_t>(o);
    return static_cast<coresets::SensitivityMethod>(i);
}

dist::DissimilarityMeasure obj2m(py::object o) {
    if(py::isinstance<py::str>(o))
        return obj2m(py::cast<py::str>(o));
    if(py::isinstance<py::int_>(o))
        return obj2m(py::cast<py::int_>(o));
    throw std::runtime_error("Unexpected object");
}
coresets::SensitivityMethod obj2sm(py::object o) {
    if(py::isinstance<py::str>(o))
        return obj2sm(py::cast<py::str>(o));
    if(py::isinstance<py::int_>(o))
        return obj2sm(py::cast<py::int_>(o));
    throw std::runtime_error("Unexpected object");
}
