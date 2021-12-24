import minicore as mc
import itertools
import sys
import numpy as np
from collections import Counter

def hiercluster_kmeanspp(dataset, radix=5, msr="JSD", n_local_trials=2, lspp=0, maxiter=3, optimize_centers=True, prior=1e-5, hierdepth=0, mbsize=-1):
    kmo = mc.kmeanspp(dataset, k=radix, msr=msr, n_local_trials=n_local_trials, lspp=lspp, prior=prior)
    centers, assignments, costs = kmo
    if optimize_centers:
        out = mc.hcluster(dataset, centers=centers, maxiter=maxiter, mbsize=mbsize)
        centers = out['centers']
        assignments = out['asn']
        costs = out['costs']
    # print(centers)
    base = {"dataset": dataset, "radix": radix, "msr": msr, "centerids": centers,"asn": assignments, "costs:": costs, "hierdepth": hierdepth}
    if isinstance(centers, np.ndarray) and centers.ndim == 1:
        base["centers"] = dataset[centers].copy()
    else:
        base["centers"] = centers
    children = {}
    # print(base)
    idlabelmap = mc.get_counthist(assignments)
    idc = {k: len(v) for k, v in idlabelmap.items()}
    # print("labelmap: ", idc)
    for k, v in idlabelmap.items():
        if len(v) > radix:
            subdataset = dataset[v].copy()
            children[k] = hiercluster_kmeanspp(subdataset, radix=radix, msr=msr, n_local_trials=n_local_trials, lspp=lspp, prior=prior, optimize_centers=optimize_centers, maxiter=maxiter, hierdepth=hierdepth + 1, mbsize=mbsize)
    base['children'] = children
    return base


if __name__ == "__main__":
    dataset = np.random.rand(102400).reshape(-1, 16)
    print(dataset.shape)
    radix = 25
    out = hiercluster_kmeanspp(dataset, radix=radix, msr="SQRL2")
    print("Base clustering has %d labels" % radix)
    subsets = list(out['children'].items())
    for k, v in subsets:
        print(f"Subset for subset {k} has {v['dataset'].shape}-shaped dataset")
    keys = [k for k, v in subsets]
    vals = [v for k, v in subsets]
    subsubsets = list(itertools.chain.from_iterable(list(x['children'].items()) for x in vals))
    subkeys = [k for k, v in subsubsets]
    subvals = [v for k, v in subsubsets]
    #print(subsubsets[0])
    for subk, v in zip(subkeys, subvals):
        print(f"Subset for subsub label {subk} has {v['dataset'].shape}-shaped dataset")
