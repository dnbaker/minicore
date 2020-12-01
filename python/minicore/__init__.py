import pyminicore
from pyminicore import *
from . import constants
from .constants import CSR as csr_tuple, KMCRSV
from pyminicore import SparseMatrixWrapper as smw
import numpy as np

cluster_from_centers = pyminicore.cluster

def spctrlist2mat(centertups, nc):
    import scipy.sparse as sp
    return sp.vstack([sp.csr_matrix((x[0],[0] * len(x[0]), [0, len(x[0])]), shape=[1, nc]) for x in centertups])
