from collections import namedtuple
SENSDICT = {"BFL": 0, "FL": 1, "LFKF": 2, "VX": 3, "LBK": 4}
CSR = namedtuple("CSR", ["data", "indices", "indptr", "shape", "nnz"])

KMCRSV = 4294967295  # -1u, asking for reservoir kmeans++ sampling.

__all__ = ["SENSDICT", "CSR", "KMCRSV"]
