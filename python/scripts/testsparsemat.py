import minicore
import scipy.sparse as sp
import numpy  as np

mat = sp.lil_matrix((1000, 100))
vals = np.random.randint(0, 100000, size=(1000,))
for row, col in zip(vals // 100, vals % 100):
    mat[row, col] = 1

mat = sp.csr_matrix(mat)
print(mat.shape)
print(mat.nnz)
assert hasattr(mat, "asformat")

smw = minicore.SparseMatrixWrapper(mat)
smw.emit()
smw.transpose_()
smw.emit()
print(str(smw))
