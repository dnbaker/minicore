import numpy as np
from scipy.io import mmread
import scipy.sparse as sp
import minicore as mc

'''Loads the DropViz experiment dataset,
   ensures that the types are correct and the sizes match
'''

PATH = "/net/langmead-bigmem-ib.bluecrab.cluster/storage/dnb/data/10xdata/berger_data/pbmc/68k/matrix.mtx"


mat = mmread(PATH)
mat.data = mat.data.astype(np.float32)  # max: 260
mat.indices = mat.indices.astype(np.uint32)
mat.indptr = mat.indptr.astype(np.uint32)
pbmc_mat = sp.csr_matrix((data, indices, indptr), shape)

__all__ = ["pbmc_mat"]
