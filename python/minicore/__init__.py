import pyfgc
from pyfgc import *
from . import constants
from .constants import CSR as csr_tuple, KMCRSV
from pyfgc import SparseMatrixWrapper as smw


def spctrlist2mat(centertups, nc):
    return sp.vstack([sp.csr_matrix((x[0],[0] * len(x[0]), [0, len(x[0])]), shape=[1, nc]) for x in centertups])
