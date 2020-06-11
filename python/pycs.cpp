#include "pyfgc.h"

void init_coreset(py::module &m) {                                                                  
    py::class_<CSType>(m, "CoresetSampler")                                                         
    .def(py::init<>())                                                                              
    .def("make_sampler", [](                                                                        
        CSType &cs, size_t ncenters, py::array costs, INA assignments, py::object weights, uint64_t seed, int sens_)
    {                                                                                               
        const auto sens(static_cast<minocore::coresets::SensitivityMethod>(sens_));                 
        py::buffer_info buf1 = costs.request();                                                     
        const uint32_t *asnp = (const uint32_t *)assignments.request().ptr;                         
        if(buf1.ndim != 1) throw std::runtime_error("buffer must have one dimension (reshape if necessary)");
        float *wp = nullptr;                                                                        
        if(auto p(pybind11::cast<FNA>(weights)); p)                                                 
            wp = static_cast<float *>(p.request().ptr);                                             
        if(py::isinstance<py::array_t<float>>(costs)) {                                             
            cs.make_sampler(ncenters, costs.shape(0), (float *)buf1.ptr, asnp, wp, seed, sens);     
        } else {                                                                                    
            cs.make_sampler(ncenters, costs.shape(0), (double *)buf1.ptr, asnp, wp, seed, sens);    
        }                                                                                           
    },                                                                                              
    "Generates a coreset sampler given a set of costs, assignments, and, optionally, weights. This can be used to generate an index coreset",
    py::arg("ncenters"), py::arg("costs"), py::arg("assignments"),                                  
    py::arg("weights") = py::cast<py::none>(Py_None), py::arg("seed") = 13, py::arg("sens")=0       
    ).def("get_probs", [](CSType &cs) {                                                             
        py::array_t<float> ret(cs.np_);                                                             
        std::copy(cs.probs_.get(), cs.probs_.get() + cs.np_, (float *)ret.request().ptr);           
        return ret;                                                                                 
    }, "Create a numpy array of sampling probabilities");                                           
}
