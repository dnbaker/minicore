from collections import Counter
from scipy.io import mmread, mmwrite
import sys
import itertools


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
                ret.append((idm[f], fid))
    return ret


class FeatureMap:
    def __init__(self, n: int, fromto):
        self.n = n
        self.cvt = {x: y for x, y in fromto}
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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", '-m', type=int, default=2)
    ap.add_argument("paths", nargs='*')
    ap = ap.parse_args()
    loaded_mats = []
    fms, features = select_features(ap.paths, min_occ_count=ap.min_count)


__all__ = ["FeatureMap", "get_ids", "get_id_map", "get_counts", "get_selected_ids", "xopen", "itertools", "mmread", "mmwrite", "select_features"]
