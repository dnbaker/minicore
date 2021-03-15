import numpy as np
import minicore as mc

def variance(mat):
    if isinstance(mat, np.ndarray):
        return np.var(mat, axis=0)
    elif not isinstance(mat, mc.SparseMatrixWrapper):
        mat = mc.SparseMatrixWrapper(mat)
    return mat.variance(kind=0)
