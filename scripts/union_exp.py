from collections import Counter
import numpy as np
from scipy.io import mmread, mmwrite
import scipy.sparse as sp
import sys
import itertools
import minocore


def xopen(x):
    if x.endswith(".xz"):
        import lzma
        return lzma.open(x)
    elif x.endswith(".gz"):
        import gzip
        return gzip.open(x)
    else:
        return open(x)
    


def get_ids(x):
    return [x for x, y in Counter(map(lambda x: x.split()[1], xopen(x))).items() if y == 1]


def get_counts(x):
    return Counter(map(lambda x: x.split()[1], xopen(x)))


def get_id_map(x):
    ret = {}
    for line in xopen(x):
        l = line.split()
        ret[l[1]] = int(l[0])
    return ret


def get_selected_ids(idc, idm, features):
    ret = []
    for fid, f in enumerate(features):
        if f in idm:
            if idc[f] == 1:
                ret.append((fid, idm[f]))
    return ret


class FeatureMap:
    def __init__(self, n: int, fromto):
        self.n = n
        self.cvt = {x: y for x, y in fromto}
        kvs = sorted(self.cvt.items())
        self.keys = np.array([x[0] for x in kvs], dtype=np.uint64)
        self.values = np.array([x[1] for x in kvs], dtype=np.uint64)
        self.nzsource = set(self.cvt.keys())
        self.nzdest = set(self.cvt.values())
        assert len(self.nzsource) == len(self.cvt)
        assert len(self.nzdest) == len(self.cvt)

    def __str__(self):
        return f"FeatureMap, mapping {self.n} features in original space to final space"


def select_features(genefnames, matrixfnames=None, min_occ_count=2):
    if matrixfnames is None:
        matrixfnames = list(map(lambda x: x.replace("genes.tsv", "matrix.mtx"), genefnames))
    assert len(genefnames) == len(matrixfnames)
    assert all(map(lambda x: x is not None, genefnames + matrixfnames))
    counts = list(map(get_counts, genefnames))
    nbc = [len(list(xopen(x))) for x in genefnames]
    gene_lists = list(map(get_ids, genefnames))
    idm = list(map(get_id_map, genefnames))
    features = sorted(x for x, y in Counter(itertools.chain.from_iterable(gene_lists)).items() if y >= min_occ_count)
    f2id = dict(zip(features, range(len(features))))
    ids = [get_selected_ids(c, idmap, features) for c, idmap in zip(counts, idm)]
    return [FeatureMap(nbc, pairs) for nbc, pairs in zip(nbc, ids)], features


def remap_mat(mat, fm, fl):
    nf = len(fl)
    mat = sp.coo_matrix(mat)
    ret = sp.coo_matrix(shape=(mat.shape[0], nf))


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp
    import minocore
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", '-m', type=int, default=2)
    ap.add_argument("paths", nargs='*')
    ap.add_argument("--prefix", "-p", type=str)
    ap = ap.parse_args()
    loaded_mats = []
    fms, features = select_features(ap.paths, min_occ_count=ap.min_count)
    matrixpaths = [x.replace("genes.tsv.xz", "matrix.mtx").replace("genes.tsv", "matrix.mtx") for x in ap.paths]
    with mp.Pool(min(4, len(matrixpaths))) as pool:
        matrices = pool.map(mmread, matrixpaths)
    matrices = list(map(lambda x: x.T, matrices))
    r, c, dat, shape = minocore.merge(matrices, fms, features)
    megamat = sp.coo_matrix((r, c, dat), shape=shape)
    megamat.row.tofile(prefix + ".row")
    megamat.col.tofile(prefix + ".col")
    megamat.data.tofile(prefix + ".coodata")
    megamat = sp.csr_matrix(megamat)
    megamat.indices.tofile(prefix + ".indices")
    megamat.indptr.tofile(prefix + ".indptr")
    megamat.data.tofile(prefix + ".data")
    megamat.shape.tofile(prefix + ".shape")
    print(megamat.shape)
    


__all__ = ["FeatureMap", "get_ids", "get_id_map", "get_counts", "get_selected_ids", "xopen", "itertools", "mmread", "mmwrite", "select_features", "np"]
