import sys
import os

from selexp import *


def fn2type(x):
    c1 = 'CLUSTER_1_ITER'
    clust = 'CLUSTER'
    grdy = 'GREEDY'
    if c1 in x:
        return c1
    elif clust in x:
        return clust
    elif grdy in x:
        if 'OUTLIERS' in x:
            return ('OUTLIERS', float(x.split("OUTLIERS_")[1].split("_")[0]))
        return grdy
    if 'D2' in x:
        return 'D2'
    raise RuntimeError("Not Expected: anything else")


def fn2measure(fn):
    return sorted(filter(lambda x: x in fn, MEASURES),
                  key=lambda x: -len(x))[0]


def load_centers(path):
    return list(map(int, map(str.strip, open(path))))


def make_table(paths):
    assert all(map(os.path.isfile, paths))
    raise NotImplementedError("make_table")


def centers2res(ctrs, trueset, samples=None):
    if samples is None:
        samples = list(range(1, len(ctrs) + 1))
    ts = trueset
    if not isinstance(ts, set):
        ts = set(trueset)
    return list(map(lambda x_i: len(ts & set(ctrs[:x_i])), samples)), samples


def getarrs(dat):
    arrs = list(map(np.array, dat['res']['scores']))


def gather_data(files, trueset, samples=None):
    types, measures, centers = (list(map(f, files)) for f in
                                (fn2type, fn2measure, load_centers))
    trueset = set(trueset)
    res = list(map(lambda x: centers2res(x, trueset, samples=samples),
                   centers))
    return {"types": types, "measures": measures, "centers": centers,
            "res": {
                "scores": res, "samples": samples
            }}
