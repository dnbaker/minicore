from union_exp import *
import minocore
from glob import glob
from minocore import merge
import scipy.sparse as sp
from scipy.io import mmread, mmwrite

def _ft(x):
    tmp = {np.int64: "i64", np.uint64: "u64", np.int32: "i32", np.uint32: "u32", np.double: "f64", np.float: "f32"}
    if x in tmp:
        return tmp[x]

    lines = '''
 np.dtype("int64")	 "i64"
 np.dtype("u64")	 "u64"
 np.dtype("int32")	 "i32"
 np.dtype("uint32")	 "u32"
 np.dtype("double")	 "f64"
 np.dtype("float32")	 "f32"'''.strip().split('\n')
    for line in lines:
        toks = line.strip().split()
        if x is eval(toks[0]): return toks[1]
    raise Exception("Failed")
    


def write_csr(mat, pref):
    mat.indices.tofile(f"{pref}.{_ft(mat.indices)}.indices")
    mat.indptr.tofile(f"{pref}.{_ft(mat.indptr)}.indptr")
    mat.data.tofile(f"{pref}.{_ft(mat.data)}.data")
    shape = np.array(mat.shape).astype(np.uint64)
    shape.tofile(f"{pref}.{_ft(shape)}.shape")


def main():
    pref = "prefix"
    if sys.argv[1:]:
        pref = sys.argv[1]
    paths = glob("*/genes.tsv")
    dirs = ["/".join(x.split("/")[:-1]) for x in paths]
    mats = list(map(sp.coo_matrix, map(mmread, map(lambda x: x.replace("genes.tsv", "matrix.mtx"), paths)))) 
    print([x.shape for x in mats])
    print("Total runs: " + str(sum(x.shape[0] for x in mats)))
    print("total nnz: " + str(sum(x.nnz for x in mats)))
    
    fmaps, feat, fidmap = select_features(paths, min_occ_count=2)
    fids = list(range(len(feat)))
    #indices_to_keep = [get_indices_to_keep(mat, path, feat, fidmap) for mat, path in zip(mats, paths)]
    rows, cols, dat, shape = minocore.merge(mats, fmaps, feat)
    print("cols", np.max(cols), cols.dtype, np.argmax(cols))
    print("rows", np.max(rows))
    
    nr = sum(x.shape[0] for x in mats)
    
    mat = sp.csr_matrix(sp.coo_matrix((dat, (rows, cols)), shape=shape))
    write_csr(mat, pref)


if __name__ == "__main__":
    import sys
    sys.exit(main())
