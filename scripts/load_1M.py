import numpy as np
import scipy.sparse as sp
import minicore as mc

'''Loads the 1.3M cell experiment dataset,
   ensures that the types are correct and the sizes match
'''

PREFIX = "/net/langmead-bigmem-ib.bluecrab.cluster/storage/dnb/data/10xdata/1Mneurons/1M."

def load_files(pref):
    data = np.fromfile(pref + "data.file", dtype=np.uint8).view(np.uint32)
    indices = np.fromfile(pref + "indices.file", dtype=np.uint8).view(np.uint64).astype(np.uint32)
    indptr = np.fromfile(pref + "indptr.file", dtype=np.uint8).view(np.uint64).astype(np.uint32)
    shape = np.fromfile(pref + "shape.file", dtype=np.uint8).view(np.uint32)[::-1]
    return (data, indices, indptr, shape)


data, indices, indptr, shape = load_files(PREFIX)
million_mat = sp.csr_matrix((data, indices, indptr), shape)

__all__ = ["million_mat"]
