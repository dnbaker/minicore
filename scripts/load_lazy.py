#from load_dropviz import zeisel_cns_mat
#from load_cao import cao_mat
#from load_pbmc import pbmc_mat
#from load_293t import t293_mat
#from load_1M import million_mat

def loadcao4m():
    import numpy as np
    pref = "/net/langmead-bigmem-ib.bluecrab.cluster/storage/dnb/data/10xdata/4MEXP"
    dat = np.fromfile(pref + "/cao4m.data.f.npy", dtype=np.float32)
    idx = np.fromfile(pref + "/cao4m.indices.u32.npy", dtype=np.uint32)
    ip = np.fromfile(pref + "/cao4m.indptr.u32.npy", dtype=np.uint32)
    shape = np.fromfile(pref + "/cao4m.shape.u32.npy", dtype=np.uint32)
    import scipy.sparse as sp
    return sp.csr_matrix((dat, idx, ip), shape=shape)

def getmat(name):
    if name == "zeisel":
        from load_dropviz import zeisel_cns_mat as ret
    elif name == 'cao':
        from load_cao import cao_mat as ret
    elif name == 'pbmc':
        from load_pbmc import pbmc_mat as ret
    elif name == '293t':
        from load_293t import t293_mat as ret
    elif name == "1.3M":
        from load_1M import million_mat as ret
    elif name == 'cao4m':
        ret = loadcao4m()
    else:
        raise RuntimeError("Not found: name")
    return ret

exp_loads = {
    "cao": lambda: getmat("cao"),
    "zeisel": lambda: getmat("zeisel"),
    "293t": lambda: getmat("293t"),
    "pbmc": lambda: getmat("pbmc"),
    "1.3M": lambda: getmat("1.3M"),
    "cao4m": lambda: getmat('cao4m')
}

ordering = ['293t', 'pbmc', 'zeisel', 'cao', '1.3M', 'cao4m']

__all__ = ["ordering", "exp_loads"]
