import pyminicore
from pyminicore import *
from .util import constants
from .util.constants import CSR as csr_tuple, KMCRSV
if hasattr(pyminicore, "SparseMatrixWrapper"):
    from pyminicore import SparseMatrixWrapper as smw
import numpy as np
from .util.compute_variance import variance
from .util import hvg


cluster_from_centers = pyminicore.hcluster

def ctrs2sp(centertups, nc):
    import scipy.sparse as sp
    return sp.vstack([sp.csr_matrix((x[0],[0] * len(x[0]), [0, len(x[0])]), shape=[1, nc]) for x in centertups])

geometric_median = pyminicore.geomed


def hcluster(matrix, centers, *, prior=0., msr=2, weights=None,
             eps=1e-10, maxiter=1000, mbsize=-1, ncheckins=-1,
             with_rep=False):
    """
    def hcluster(matrix, centers, *, prior=0., msr=2, weights=None,
                 eps=1e-10, maxiter=1000, mbsize=-1, ncheckins=-1):
    This is a wrapper function that automatically calls the correct pyminicore hcluster
    It uses Expectation Maximization to minimize the cost of the dataset,
    but it requires an initial set of centers for an experiment.
    To get that initial set, look at kmeanspp for most uses, though you might want to try
    greedy_select, which uses farthest-point sampling.

    Inputs:
        matrix:
            This can be scipy.sparse.csr_matrix, minicore.CSparseMatrix,
            or minicore.csr_tuple, or a 2-d numpy ndarray.
            The dataset to be clustered.
        centers:
            This can be a dense 2-d numpy array or a list of integers or a numpy array
            of an integral type indicating rows to use an initial centers for clustering.

    Keyword Arguments:
        prior:
            Prior/Pseudocount adjustment.
            Defaults to 0, although a small adjustment may be made for numerical stability.
            Raising this number pushes points closer together for many distances.
        msr:
            Distance measure. Default is 2 (squared L2). For a full list, see the end of this doc-string.
            You may be interested in the KL divergence (5/MKL), Jensen-Shannon Divergence (JSD), TVD, or Hellinger.
            Some of these yield imaginary results with negative numbers; minicore expects nonnegative-valued observations.
        weights:
            None by default; Provide a numpy array of weights for each point to enable weighted clustering
        eps:
            Tolerance for the k-means convergence algorithm; only actively used for full k-means; (not mini-batch)
        maxiter:
            Default 1000, maximum number of outer loops to iterate
        mbsize:
            If this is not defaulted, then minibatch k-means clustering is performed
            with the argument as minibatch size. A smaller number provides faster clustering
            with lower quality solutions.
        ncheckins:
            Set ncheckins to determine how many inner loops are taken between outer loop steps in minibatch k-means.
            Raising this number may increase or decrease speed.

    #Distance Table -- Numeric ID\tKey\tDescription
    0	L1	L1 distance
    1	L2	L2 distance
    2	SQRL2	Squared L2 Norm
    3	JSM	Jensen-Shannon Metric, known as S2JSD and the Endres metric, for Poisson and Multinomial models, for which they are equivalent
    4	JSD	Jensen-Shannon Divergence for Poisson and Multinomial models, for which they are equivalent
    5	MKL	Multinomial KL divergence
    6	HELLINGER	Hellinger Distance: sqrt(sum((sqrt(x) - sqrt(y))^2))/2
    7	BHATTACHARYYA_METRIC	Bhattacharyya metric: sqrt(1 - BhattacharyyaSimilarity(x, y))
    8	BHATTACHARYYA_DISTANCE	Bhattacharyya distance: -log(dot(sqrt(x) * sqrt(y)))
    9	TOTAL_VARIATION_DISTANCE	Total Variation Distance: 1/2 sum_{i in D}(|x_i - y_i|)
    10	LLR	Log-likelihood Ratio under the multinomial model
    11	REVERSE_MKL	Reverse Multinomial KL divergence
    12	UWLLR	Unweighted Log-likelihood Ratio. This is effectively the Generalized Jensen-Shannon Divergence with lambda parameter corresponding to the fractional contribution of counts in the first observation. This is symmetric, unlike the G_JSD, because the parameter comes from the counts.
    13	ITAKURA_SAITO	Itakura-Saito divergence, a Bregman divergence [sum((a / b) - log(a / b) - 1 for a, b in zip(A, B))]
    14	REVERSE_ITAKURA_SAITO	Reversed Itakura-Saito divergence, a Bregman divergence
    15	COSINE_DISTANCE	Cosine distance: arccos(\frac{A \cdot B}{|A|_2 |B|_2}) / pi
    17	COSINE_SIMILARITY	Cosine similarity: \frac{A \cdot B}{|A|_2 |B|_2}
    23	SYMMETRIC_ITAKURA_SAITO	Symmetrized Itakura-Saito divergence. IS is a [sum((a / b) - log(a / b) - 1 for a, b in zip(A, B))], while SIS is .5 * (IS(a, (a + b) / 2) + IS(b, (a + b) / 2)), analogous to JSD
    24	RSYMMETRIC_ITAKURA_SAITO	Reversed symmetrized Itakura-Saito divergence. IS is a [sum((a / b) - log(a / b) - 1 for a, b in zip(A, B))], while SIS is .5 * (IS((a + b) / 2, a) + IS((a + b) / 2, b)), analogous to JSD
    25	SRLRT	Square root of LRT, the log likelihood ratio test; related to the JSM and Generalized JSD
    26	SRULRT	Square root of UWLLR, unweighted log likelihood ratio test; related to the JSM and Generalized JSD
    """
    import scipy.sparse as sp
    if isinstance(matrix, sp.csr_matrix) or isinstance(matrix, csr_tuple):
        matrix = mc.CSparseMatrix(matrix)
    argmat = matrix
    if isinstance(argmat, sp.csr_matrix) or isinstance(argmat, csr_tuple):
        argmax = mc.CSparseMatrix(argmat)
    return pyminicore.hcluster(argmat, centers=centers, prior=prior, msr=msr, weights=weights, eps=eps,
                               maxiter=maxiter, mbsize=mbsize, ncheckins=ncheckins,
                               with_rep=with_rep)

#  This function provides a single starting point for clustering start-to-finish
def cluster(data, *, msr, k, prior=0., seed=0, nmkc=0,
            ntimes=1, lspp=0, use_exponential_skips=False,
            n_local_trials=1, weights=None, mbsize=-1, clustereps=1e-4,
            temp=-1., cs=False, with_rep=True, outpref="mc.cluster.output",
            maxiter=50):
    soft = temp > 0.  # Enable soft clustering by setting temperature
    if isinstance(data, csr_tuple):
        mcdata = CSparseMatrix(data)
    else:
        mcdata = data
    ids, asn, costs = pyminicore.kmeanspp(data, msr=msr, k=k,
                                          prior=prior, seed=seed,
                                          ntimes=ntimes, lspp=lspp, expskips=use_exponential_skips,
                                          n_local_trials=n_local_trials, weights=weights)
    if soft:
        try:
            return scluster(mcdata, centers=ids, msr=msr, prior=prior, weights=weights, temp=temp, maxiter=maxiter,
                            mbn=mbn, savepref=outpref)
        except:
            raise NotImplementedError("Soft clustering not supported")
    else:
        return hcluster(data, centers=ids, prior=prior, msr=msr, weights=weights, eps=clustereps, maxiter=maxiter,
                        with_rep=with_rep, cs=cs)

'''
__all__ = ["cluster", "scluster", "hcluster", "hvg", "variance",
           "cstr2sp", "cluster_from_centers", "geomed", "geometric_median",
           "CSparseMatrix", "SparseMatrixWrapper", "smw", "csr_tuple", "KMCRSV",
           "constants", "pyminicore", "greedy_select", "d2_select", "SumOpts", "meas2str", "meas2dict",
           "usable_measures", "valid_measures", "display_measures", "mdict", "Threading", "set_num_threads", "get_num_threads"]
'''
