import numpy as np
import minicore as mc


def variance(mat):
    '''
        Input: np.ndarray or mc.SparseMatrixWrapper or mc.CSparseMatrix
        Returns: np.ndarray, one-dimensional, column-wise variance
    '''
    if isinstance(mat, np.ndarray):
        return np.var(mat, axis=0)
    elif not isinstance(mat, mc.SparseMatrixWrapper):
        mat = mc.SparseMatrixWrapper(mat)
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
    if isinstance(mat, np.ndarray):
        return (mat[:, topnids].copy(), topnids, topnv)
    elif not isinstance(mat, mc.SparseMatrixWrapper):
        mat = mc.SparseMatrixWrapper(mat)
    return (mat.submatrix(columnsel=topnids), topnids, topnv)


__all__ = ["variance", "hvg"]
