from load_dropviz import zeisel_cns_mat
from load_cao import cao_mat
from load_pbmc import pbmc_mat
from load_293t import t293_mat
from load_1M import million_mat

all_experiments = {"cao": cao_mat, "zeisel": zeisel_cns_mat, "pbmc": pbmc_mat, "293t": t293_mat, "1.3M": million_mat}

ordering = ['293t', 'pbmc', 'zeisel', 'cao', '1.3M']

__all__ = ["zeisel_cns_mat", "cao_mat", "pbmc_mat", "t293_mat", "all_experiments", "million_mat", "ordering"]
