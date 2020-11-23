import numpy as np
import scipy.sparse as sp
import minicore as mc

'''Loads the DropViz experiment dataset,
   ensures that the types are correct and the sizes match
'''

PREFIX = "/net/langmead-bigmem-ib.bluecrab.cluster/storage/dnb/data/10xdata/berger_data/mouse_brain/zeisel/filteredstacked."

def load_files(pref):
    data = np.fromfile(pref + "data.file", dtype=np.uint8).view(np.uint64).astype(np.uint32)
    indices = np.fromfile(pref + "indices.file", dtype=np.uint8).view(np.uint32)
    indptr = np.fromfile(pref + "indptr.file", dtype=np.uint8).view(np.uint32)
    shape = np.fromfile(pref + "shape.file", dtype=np.uint8).view(np.uint32)
    return (data, indices, indptr, shape)


data, indices, indptr, shape = load_files(PREFIX)
zeisel_cns_mat = sp.csr_matrix((data, indices, indptr), shape)

__all__ = ["zeisel_cns_mat"]
