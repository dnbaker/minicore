#include "pyfgc.h"


py::ssize_t threadgetter() {
    py::ssize_t ret = 1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        ret = omp_get_num_threads();
    }
#endif
    return ret;
}
void threadsetter(py::ssize_t x) {
    if(x > 0) omp_set_num_threads(x);
}
struct OMPThreadNumManager {
    OMPThreadNumManager(int nthreads=-1) {set(nthreads);}
    void set(py::ssize_t nthreads) const {threadsetter(nthreads);}
    py::ssize_t get() const {return threadgetter();}
};

void init_omp_helpers(py::module &m) {
    m.def("set_num_threads", threadsetter);
    m.def("get_num_threads", threadgetter);
    py::class_<OMPThreadNumManager>(m, "Threading").def(py::init<>()).def(py::init<py::ssize_t>())
    .def_property("nthreads", &OMPThreadNumManager::get, &OMPThreadNumManager::set)
    .def_property("p", &OMPThreadNumManager::get, &OMPThreadNumManager::set);
}
