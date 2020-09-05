#include "pyfgc.h"

void init_omp_helpers(py::module &m) {
    m.def("set_num_threads", [](Py_ssize_t x) {
        OMP_ELSE(omp_set_num_threads(x), );
    });
    m.def("get_num_threads", []() {
        int ret = 1;
#ifdef _OPENMP
        #pragma omp parallel
        {
            #pragma omp single
            ret = omp_get_num_threads();
        }
#endif
        return ret;
    });
}
