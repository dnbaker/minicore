import numpy as np
import minicore as mc
import scipy.sparse as sp


def variance(mat):
    '''
        Input: np.ndarray or mc.SparseMatrixWrapper or mc.CSparseMatrix
        Returns: np.ndarray, one-dimensional, column-wise variance
    '''
    if isinstance(mat, np.ndarray):
        return np.var(mat, axis=0)
    elif isinstance(mat, sp.csr_matrix):
        asq = mat.copy()
        asq.data *= asq.data
        return np.array(asq.mean(axis=0) - np.square(mat.mean(axis=0))).reshape(asq.shape[1])
    elif not isinstance(mat, mc.SparseMatrixWrapper):
        try:
            mat = mc.SparseMatrixWrapper(mat)
        except RuntimeError as e:
            mat = sp.csr_matrix((mat.data, mat.indices, mat.indptr), shape=mat.shape).astype(np.float32)
            return variance(mat)
    return mat.variance(kind=0)


def hvg(mat, n=500, topnorm=False):
    '''
        Input:
            mat - np.ndarray or mc.SparseMatrixWrapper or mc.CSparseMatrix
            n = 500 - Number of features to keep
            topnorm = False - Whether to normalize variances by
                              the means of the features;
        Output:
            tuple: (matrix, ids, variances)
    '''
    vs = variance(mat)

    def sortkey(x):
        return -x[1]
    if topnorm:
        means = np.mean(mat, axis=0)
        topn = sorted(enumerate(vs / means), key=sortkey)[:n]
    else:
        topn = sorted(enumerate(vs), key=sortkey)[:n]
    topnids = np.sort(np.array([x[0] for x in topn]))
    topnv = np.array([x[1] for x in topn])
    if isinstance(mat, np.ndarray) or isinstance(mat, sp.csr_matrix):
        return (mat[:, topnids].copy(), topnids, topnv)
    elif not isinstance(mat, mc.SparseMatrixWrapper):
        try:
            mat = mc.SparseMatrixWrapper(mat)
        except RuntimeError as e:
            csrmat = sp.csr_matrix((mat.data, mat.indices, mat.indptr), shape=mat.shape).astype(np.float32)
            mat = mc.SparseMatrixWrapper(csrmat)
    return (mat.submatrix(columnsel=topnids), topnids, topnv)


__all__ = ["variance", "hvg"]
