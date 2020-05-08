#include "blaze/math/DynamicMatrix.h"
#include "aesctr/wy.h"
#include "minocore/clustering.h"

using namespace minocore;

template<typename FT>
blaze::DynamicMatrix<FT> parse_file(std::string path, unsigned *num_clusters) {
    std::ifstream ifs(path);
    std::string line;
    if(!std::getline(ifs, line)) throw 1;
    size_t nr = std::atoi(line.data());
    size_t nc = std::atoi(std::strchr(line.data(), '/') + 1);
    *num_clusters = std::atoi(std::strchr(std::strchr(line.data(), '/') + 1, '/') + 1);
    blaze::DynamicMatrix<FT> ret(nr, nc);
    size_t row_index = 0;
    while(std::getline(ifs, line)) {
        auto r = row(ret, row_index++);
        char *ptr = line.data();
        for(size_t col_index = 0;col_index < nc;r[col_index++] = std::strtod(ptr, &ptr));
    }
    assert(row_index == nr);
    return ret;
}


int main(int argc, char **argv) {
    int ret = 0;
    unsigned k;
    std::string inpath = "random.out";
    if(argc > 1) inpath = argv[1];
    auto pointmat = parse_file<float>(inpath, &k);

    std::cerr << "Parsed matrix of " << pointmat.rows() << " rows and "
              << pointmat.columns() << " columns, with k = " << k << " clusters\n";
    auto jsdapp = make_probdiv_applicator(pointmat, blz::SQRL2);
    std::cerr << "Made probdiv applicator\n";
    auto clusterdata = clustering::perform_clustering<clustering::HARD, clustering::EXTRINSIC>(jsdapp, k);
    shared::flat_hash_map<uint32_t, shared::flat_hash_set<uint32_t>> labels, clabels;
    std::ifstream ifs(inpath + ".labels.txt");
    size_t lno = 0;
    for(std::string l; std::getline(ifs, l);) {
        labels[std::atoi(l.data())].insert(lno++);
    }
    auto &asn = std::get<1>(clusterdata);
    for(size_t i = 0; i < asn.size(); ++i) {
        clabels[asn[i]].insert(i);
    }
    shared::flat_hash_set<uint32_t> sizes, csizes;
    for(const auto &l: labels) sizes.insert(l.second.size());
    for(const auto &l: clabels) csizes.insert(l.second.size());
    std::cerr << "sizes size: " << sizes.size() << '\n';
    std::cerr << "csizes size: " << csizes.size() << '\n';
    assert(sizes.size() == csizes.size() && *sizes.begin() == *csizes.begin());
    return ret;
}
