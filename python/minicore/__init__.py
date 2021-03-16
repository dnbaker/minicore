import pyminicore
from pyminicore import *
from .util import constants
from .util.constants import CSR as csr_tuple, KMCRSV
if hasattr(pyminicore, "SparseMatrixWrapper"):
    from pyminicore import SparseMatrixWrapper as smw
import numpy as np
from .util.compute_variance import variance
from .util import hvg


cluster_from_centers = pyminicore.hcluster

def ctrs2sp(centertups, nc):
    import scipy.sparse as sp
    return sp.vstack([sp.csr_matrix((x[0],[0] * len(x[0]), [0, len(x[0])]), shape=[1, nc]) for x in centertups])


#  This function provides a single starting point for clustering start-to-finish
def cluster(data, *, msr, k, prior=0., seed=0, nmkc=0,
            ntimes=1, lspp=0, use_exponential_skips=False,
            n_local_trials=1, weights=None, mbsize=-1, clustereps=1e-4,
            temp=-1., cs=False, with_rep=True, outpref="mc.cluster.output",
            maxiter=50):
    soft = temp > 0.  # Enable soft clustering by setting temperature
    if isinstance(data, csr_tuple):
        mcdata = CSparseMatrix(data)
    else:
        mcdata = data
    ids, asn, costs = pyminicore.kmeanspp(data, msr=msr, k=k,
                                          prior=prior, seed=seed,
                                          ntimes=ntimes, lspp=lspp, expskips=use_exponential_skips,
                                          n_local_trials=n_local_trials, weights=weights)
    if soft:
        try:
            return scluster(mcdata, centers=ids, msr=msr, prior=prior, weights=weights, temp=temp, maxiter=maxiter,
                            mbn=mbn, savepref=outpref)
        except:
            raise NotImplementedError("Soft clustering not supported")
    else:
        return hcluster(data, centers=ids, prior=prior, msr=msr, weights=weights, eps=clustereps, maxiter=maxiter,
                        with_rep=with_rep, cs=cs)
