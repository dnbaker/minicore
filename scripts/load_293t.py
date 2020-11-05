import numpy as np
from scipy.io import mmread
import scipy.sparse as sp
import minicore as mc

'''Loads the 293T/Jurkat spike-in experiment and
   ensures that the types are correct and the sizes match
'''

PATH = "/net/langmead-bigmem-ib.bluecrab.cluster/storage/dnb/data/10xdata/berger_data/293t_jurkat/jurkat_293t_99_1/matrix.mtx"


mat = mmread(PATH)
mat.data = mat.data.astype(np.float32)  # max: 260
mat = mat.T.tocsr()
mat.indices = mat.indices.astype(np.uint32)
mat.indptr = mat.indptr.astype(np.uint32)
t293_mat = mat

__all__ = ["t293_mat"]
