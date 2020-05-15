import h5py
import numpy as np
import sys

infname = sys.argv[1]
key = sys.argv[2] if sys.argv[2:] else "matrix"
prefix = "data" if not sys.argv[3:] else sys.argv[3]

f = h5py.File(infname, "r")
print(f.keys())
group = f[key]
for comp in ["shape", "indices", "indptr", "data"]:
    with open(prefix + '.' + comp, "w") as f:
        np.array(group[comp]).tofile(f)
