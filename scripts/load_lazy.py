#from load_dropviz import zeisel_cns_mat
#from load_cao import cao_mat
#from load_pbmc import pbmc_mat
#from load_293t import t293_mat
#from load_1M import million_mat

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
    else:
        raise RuntimeError("Not found: name")
    return ret

exp_loads = {
    "cao": lambda: getmat("cao"),
    "zeisel": lambda: getmat("zeisel"),
    "293t": lambda: getmat("293t"),
    "pbmc": lambda: getmat("pbmc"),
    "1.3M": lambda: getmat("1.3M")
}

ordering = ['293t', 'pbmc', 'zeisel', 'cao', '1.3M']

__all__ = ["ordering", "exp_loads"]
