#include "pyfgc.h"
#include "smw.h"




template<typename T>
std::vector<py::ssize_t> topindices(T *ptr, size_t nf, size_t nti) {
    std::unique_ptr<std::pair<double, py::ssize_t>[]> dat(new std::pair<double, py::ssize_t>[nti]);
    const size_t e = nf;
    size_t i;
    for(i = 0; i < nti; dat[i] = {ptr[i], i}, ++i);
    std::make_heap(dat.get(), dat.get() + nti, std::greater<std::pair<double, py::ssize_t>>());
    for(;i < e; ++i) {
        if(ptr[i] > dat.get()->first) {
            std::pop_heap(dat.get(), dat.get() + nti, std::greater<std::pair<double, py::ssize_t>>());
            dat[nti - 1] = {ptr[i], i}; 
            std::push_heap(dat.get(), dat.get() + nti, std::greater<std::pair<double, py::ssize_t>>());
        }   
    }   
    for(size_t i = 0; i < nf; ++i) std::fprintf(stderr, "%g/%zu\n", dat[i].first, dat[i].second);
    std::vector<py::ssize_t> indices(nf);
    std::transform(dat.get(), &dat[nf], indices.data(), [](auto x) {return x.second;});
    std::sort(indices.data(), &indices[nf]);
    return indices;
}

void init_hvg(py::module &m) {
    m.def("hvf", [](py::object item, py::object nf) -> py::object {
        py::ssize_t actualnf = py::cast<py::ssize_t>(nf);
        if(py::isinstance<py::array>(item)) {
            auto iinf = py::cast<py::array>(item).request();
            auto dt = standardize_dtype(iinf.format)[0];
            if(iinf.ndim != 2) throw std::invalid_argument("hvf must be performed on a 2-d numpy array, a scipy sparse matrix, or a minicore sparse matrix");
            if(actualnf > iinf.shape[1]) throw std::invalid_argument("hvf has fewer features than requested");
            blaze::DynamicVector<double> variance(iinf.shape[1]);
            switch(dt) {
                case 'I': {
                    blaze::CustomMatrix<uint32_t, blaze::unaligned, blaze::unpadded> cm((uint32_t *)iinf.ptr, iinf.shape[0], iinf.shape[1]);
                    variance = trans(blaze::var<blaze::columnwise>(cm));
                    break;
                }
                case 'i': {
                    blaze::CustomMatrix<int32_t, blaze::unaligned, blaze::unpadded> cm((int32_t *)iinf.ptr, iinf.shape[0], iinf.shape[1]);
                    variance = trans(blaze::var<blaze::columnwise>(cm));
                    break;
                }
                case 'd': {
                    blaze::CustomMatrix<double, blaze::unaligned, blaze::unpadded> cm((double *)iinf.ptr, iinf.shape[0], iinf.shape[1]);
                    variance = trans(blaze::var<blaze::columnwise>(cm));
                    break;
                }
                case 'f': {
                    blaze::CustomMatrix<float, blaze::unaligned, blaze::unpadded> cm((float *)iinf.ptr, iinf.shape[0], iinf.shape[1]);
                    variance = trans(blaze::var<blaze::columnwise>(cm));
                    break;
                }
                default: throw std::invalid_argument("Matrix must be composed of u32, i32, float, or double");
            }
            auto indices = topindices(variance.data(), variance.size(), actualnf);
            if(iinf.itemsize > 4) {
                py::array_t<double> ret({iinf.shape[0], actualnf});
                auto rinf = ret.request();
                if(dt != 'd') throw std::invalid_argument("Expected double or <=4byte data");
                blaze::CustomMatrix<double, blaze::unaligned, blaze::unpadded> cm((double *)rinf.ptr, iinf.shape[0], actualnf);
                cm = columns(blaze::CustomMatrix<double, blaze::unaligned, blaze::unpadded>((double *)iinf.ptr, iinf.shape[0], iinf.shape[1]), indices.data(), indices.size());
            } else {
                py::array_t<float, py::array::c_style | py::array::forcecast> castdata(py::cast<py::array>(item));
                py::array_t<float> ret({iinf.shape[0], actualnf});
                auto iinf = castdata.request();
                auto rinf = ret.request();
                blaze::CustomMatrix<float, blaze::unaligned, blaze::unpadded> cm((float *)rinf.ptr, iinf.shape[0], actualnf);
                cm = columns(blaze::CustomMatrix<float, blaze::unaligned, blaze::unpadded>((float *)iinf.ptr, iinf.shape[0], iinf.shape[1]), indices.data(), indices.size());
                return ret;
            }
        } else {
            if(py::isinstance<SparseMatrixWrapper>(item)) {
                auto mat = item.cast<SparseMatrixWrapper>();
                blaze::DynamicVector<double> variance(mat.columns());
                mat.perform([&](auto &fmat) {variance = trans(blaze::var<blaze::columnwise>(fmat));});
                std::vector<py::ssize_t> topind = topindices(variance.data(), variance.size(), actualnf);
                py::array_t<double> ret(std::vector<py::ssize_t>{py::ssize_t(mat.rows()), actualnf});
                auto rinf = ret.request();
                blaze::CustomMatrix<float, blaze::unaligned, blaze::unpadded> cm((float *)rinf.ptr, mat.rows(), actualnf);
                mat.perform([&](auto &fmat) {cm = columns(fmat, topind.data(), topind.size());});
                return ret;
            } else if(!py::hasattr(item, "indices") || !py::hasattr(item, "indptr") || !py::hasattr(item, "data")) {
                throw std::runtime_error("Expected a dense numpy matrix or a sparse scipy or minicore matrix");
            } else throw std::runtime_error("Not implemented for CSR matrices: hvf");
        }
        return py::none(); // This never happens
    });
}
