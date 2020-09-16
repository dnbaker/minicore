from collections import namedtuple
SENSDICT = {"BFL": 0, "FL": 1, "LFKF": 2, "VX": 3, "LBK": 4}
CSR = namedtuple("CSR", ["data", "indices", "indptr", "shape", "nnz"]) 

__all__ = ["SENSDICT", "CSR"]
