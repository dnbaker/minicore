#include "H5Cpp.h"
#include <cassert>
#include <memory>
//#include <h5cpp/hdf5.hpp>
#include <iostream>
#if 0
herr_t
file_info(hid_t loc_id, const char *name, const H5L_info_t *linfo, void *opdata)
{
    hid_t group;
    /*
     * Open the group using its name.
     */
    group = H5Gopen2(loc_id, name, H5P_DEFAULT);
    /*
     * Display group name.
     */
    std::cout << "Name : " << name << '\n';
    H5Gclose(group);
    return 0;
}
#endif


void write_dataset_to_fname(std::string path, std::string dskey, H5::Group &group) {
    auto dataset = group.openDataSet(dskey);
    auto nbytes = dataset.getStorageSize();
    auto nitems = dataset.getSpace().getSimpleExtentNpoints();
    std::fprintf(stderr, "nbytes: %zu. nitems: %zu. Size per item: %zu\n", nbytes, nitems, nbytes/ nitems)
    std::FILE *ofp = std::fopen(path.data(), "a+");
    auto fd = fileno(ofp);
    ::ftruncate(fd, nbytes);
    mio::mmap_sink(fd, 0, nbytes);
}



int main(int argc, char *argv[]) {
    // TODO: extract to binary file, then iterate over the file.
    std::string inpath = "5k_pbmc_protein_v3_raw_feature_bc_matrix.h5";
    if(argc > 1) inpath = argv[1];
    H5::H5File file(inpath.data(), H5F_ACC_RDONLY );
    auto group = H5::Group(file.openGroup("matrix"));
    auto shape = group.openDataSet("shape");
    auto data = group.openDataSet("data");
    auto indices = group.openDataSet("indices");
    auto indptr = group.openDataSet("indptr");
    //auto barcodes = group.openDataSet("barcodes");
    auto sid = shape.getId(), gid = group.getId(), did = data.getId(), iid = indices.getId(),
       indid = indices.getId();
#if 0
    std::cerr << "shape size: " << shape.getInMemDataSize() << '\n';
    std::cerr << "indices size: " << indices.getSpace().getSimpleExtentNpoints() << '\n';
    std::cerr << "indptr size: " << indptr.getSpace().getSimpleExtentNpoints() << '\n';
    std::cerr << "shape size: " << shape.getStorageSize() << '\n';
#endif
    hsize_t start, stop;
    shape.getSpace().getSelectBounds (&start, &stop);
    std::cerr << start << ", " << stop << '\n';
    H5::DataSpace dspace = shape.getSpace();
    hsize_t dims[2]{0};
    hsize_t rank = dspace.getSimpleExtentDims(dims, NULL);
    std::cout << "dims: " << dims[0] << ',' << dims[1] << '\n';
    assert(shape.getSpace().getSimpleExtentNpoints () == 2);
    size_t n_indices = indptr.getSpace().getSimpleExtentNpoints();
    auto indptrd = std::make_unique<uint64_t[]>(n_indices);
    indptr.read(indptrd.get(), H5::PredType::STD_I64LE);
    uint32_t shape_out[2];
    assert(shape.getIntType().getSize() == 4);
    shape.read(shape_out, H5::PredType::STD_I32LE);
    std::cout << shape_out[0] << ',' << shape_out[1] << '\n';
    //herr_t idx = H5Literate(file.getId(), H5_INDEX_NAME, H5_ITER_INC, NULL, file_info, NULL);
    //herr_t idx2 = H5Literate(group.getId(), H5_INDEX_NAME, H5_ITER_INC, NULL, file_info, NULL);
    //H5T_class_t type_class = dataset.getTypeClass();
    //std::cout << type_class << '\n';
}
