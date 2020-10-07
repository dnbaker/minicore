#include "pyfgc.h"

void init_omp_helpers(py::module &m) {
    auto threadsetter = [](Py_ssize_t x) {
#ifdef _OPENMP
         omp_set_num_threads(x);
#else
         if(x != 1) {
             std::fprintf(stderr, "set_num_threads ignored because OpenMP is not enabled\n");
         }
#endif
     };
    auto threadgetter = []() {
        int ret = 1;
#ifdef _OPENMP
        #pragma omp parallel
        {
            #pragma omp single
            ret = omp_get_num_threads();
        }
#endif
        return ret;
    };
    m.def("set_num_threads", threadsetter);
    m.def("get_num_threads", threadgetter);
}
