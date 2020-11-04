import numpy as np
from scipy.io import mmread
import scipy.sparse as sp
import pickle
import minicore as mc

'''Loads the PBMC experiment dataset,
   ensures that the types are correct and the sizes match
'''

PATH="/net/langmead-bigmem-ib.bluecrab.cluster/storage/dnb/code2/clusterdash/minocore/scripts/pbmc.bin.pkl"

pbmc_mat = pickle.load(open(PATH, "rb"))

pbmc_mat.indices = pbmc_mat.indices.astype(np.uint32)

__all__ = ["pbmc_mat"]
