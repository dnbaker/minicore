#include "H5Cpp.h"
#include "mio/single_include/mio/mio.hpp"
#include <cassert>
#include <memory>
//#include <h5cpp/hdf5.hpp>
#include <iostream>


void write_dataset_to_fname(std::string path, std::string dskey, H5::Group &group) {
    auto dataset = group.openDataSet(dskey.data());
    auto nitems = dataset.getSpace().getSimpleExtentNpoints();
    auto dt = dataset.getDataType();
    auto itemsz = dt.getSize();
    size_t nbytes = itemsz * nitems;
    std::fprintf(stderr, "[%s] nbytes: %zu. nitems: %zu. Size per item: %zu\n", dskey.data(), nbytes, size_t(nitems), size_t(itemsz));
    std::FILE *ofp = std::fopen(path.data(), "a+");
    auto fd = fileno(ofp);
    ::ftruncate(fd, nbytes);
    mio::mmap_sink ms(fd);
    uint8_t *data = (uint8_t *)ms.data();
    dataset.read((void *)data, dataset.getDataType());
    std::fclose(ofp);
}


int main(int argc, char *argv[]) {
    // TODO: extract to binary file, then iterate over the file.
    std::string inpath = "5k_pbmc_protein_v3_raw_feature_bc_matrix.h5";
    std::string outpref = "";
    if(argc > 1) inpath = argv[1];
    H5::H5File file(inpath.data(), H5F_ACC_RDONLY );
    auto group = H5::Group(file.openGroup("matrix"));
    auto shape = group.openDataSet("shape");
    assert(shape.getIntType().getSize() == 4);
    uint32_t shape_out[2];
    shape.read(shape_out, H5::PredType::STD_I32LE);
    std::cout << shape_out[0] << ',' << shape_out[1] << '\n';
    for(const auto str: {"shape", "data", "indices", "indptr"}) {
        std::string outfname = outpref + str + ".file";
        write_dataset_to_fname(outfname, str, group);
    }
}
