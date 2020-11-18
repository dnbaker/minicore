import numpy as np
import scipy.sparse as sp
import minicore as mc

'''Loads the Cao, et al. experiment dataset,
   ensures that the types are correct and the sizes match
'''

PREFIX = "/net/langmead-bigmem-ib.bluecrab.cluster/storage/dnb/data/10xdata/CAO/cao_atlas_"

def load_files(pref):
    data = np.fromfile(pref + "data.file", dtype=np.uint8).view(np.float32)
    indices = np.fromfile(pref + "indices.file", dtype=np.uint8).view(np.uint32)
    indptr = np.fromfile(pref + "indptr.file", dtype=np.uint8).view(np.uint64)
    shape = np.fromfile(pref + "shape.file", dtype=np.uint8).view(np.uint32)[::-1]
    return (data, indices, indptr, shape)


data, indices, indptr, shape = load_files(PREFIX)
cao_mat= sp.csr_matrix((data, indices, indptr), shape)

__all__ = ["cao_mat"]
