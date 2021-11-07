from pyminicore import kmeanspp as pykmpp
from pyminicore import SumOpts, CSparseMatrix, CoresetSampler
from pyminicore import pcmp, cmp
from pyminicore import SparseMatrixWrapper as smw
from pyminicore import get_counthist
import pyminicore
from .util import constants
from .util.constants import CSR as csr_tuple, KMCRSV
import numpy as np
from .util.compute_variance import variance
from .util import hvg
import scipy.sparse as sp


def cluster_from_centers(*args, **kwargs):
    from pyminicore import hcluster
    return hcluster(*args, **kwargs)

def ctrs2sp(centertups, nc):
    return sp.vstack([sp.csr_matrix((x[0],[0] * len(x[0]), [0, len(x[0])]), shape=[1, nc]) for x in centertups])

def kmeanspp(matrix, k, *, msr=2, prior=0., seed=0, ntimes=1, lspp=0, expskips=0, n_local_trials=1, weights=None):
    """
    Compute kmeans++ over the input matrix for given parameters.

        Inputs:
            Input matrix: sp.csr_matrix, minicore.CSparseMatrix, or numpy.ndarray

        Second argument (k=) can be either pyminicore.SumOpts, which is a tuple holding clustering options
                or, more commonly, an integer describing how many centers to sample.
        If this is not a SumOpts object, then the rest of the kwargs are used.

            seed [0] -- Set a seed for random sampling.
            expskips [false] -- Set to do exponential skips without simd sampling
            n_local_trials [1] -- Perform more than 1 samples per iteration during kmeans++
                                  Higher quality approximation, increased runtime cost
            ntimes  [1] -- Perform whole algorithm <argument> times, returning the solution with minimal objective cost.
            weights [None] -- provide a 1-d numpy array of weights to perform weighted kmeans++ sampling.


        Outputs:
            Tuple of size three, consisting of:
                np.ndarray[Int] - Sampled points
                np.ndarray[Int] - Assignments
                np.ndarray[Float] - Costs of each point
    """
    if isinstance(matrix, sp.csr_matrix) or isinstance(matrix, csr_tuple):
        matrix = CSparseMatrix(matrix)
    if isinstance(k, SumOpts):
        return pykmpp(matrix, k, weights=weights)
    return pykmpp(matrix, k=k, msr=msr, prior=prior, seed=seed, ntimes=ntimes,
                               lspp=lspp, expskips=expskips,
                               n_local_trials=n_local_trials,
                               weights=weights)

'''
def cmp(x, y=None, *, msr=2, prior=0., reverse=False, use_float=True):
    """
    Performs distance calculations between x and y
    If y is None, performs pairwise comparisons against itself.
    """
    if y is None:
        return pyminicore.pcmp(x, msr=msr, prior=prior, use_float=use_foat)
    if isinstance(x, sp.csr_matrix):
        x = mc.CSparseMatrix(x)
    if isinstance(y, sp.csr_matrix):
        y = mc.CSparseMatrix(y)
    return pyminicore.cmp(x, y, msr=msr, prior=prior, reverse=
'''


def hcluster(matrix, centers, *, prior=0., msr=2, weights=None,
             eps=1e-10, maxiter=1000, mbsize=-1, ncheckins=-1,
             with_rep=False, cs=False):
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

    Returns:
        Dictionary with keys --
        "asn" - np.array, uint32_t [num samples]
        "costs" - np.array, float [num samples]
        "initcost" - float, initial cost before clustering
        "finalcost" - float, final cost after clustering
        "numiter" - int, number of iterations clustering performed

        "centers" - final centers:
            If the input matrix is dense, this will be a 2-d ndarray of the centers
            Otherwise, this will be a tuple for the CSR notation of the centers as ((data, indices, indptr), shape)
            You might convert this to a SciPy CSR matrix by:
            ```
            tup, shape = ret["centers"]
            smat = scipy.sparse.csr_matrix(tup, shape=shape)
            ```

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
    if isinstance(matrix, sp.csr_matrix) or isinstance(matrix, csr_tuple):
        matrix = CSparseMatrix(matrix)
    argmat = matrix
    from pyminicore import hcluster as pmhc
    return pmhc(argmat, centers=centers, prior=prior, msr=msr, weights=weights, eps=eps,
                maxiter=maxiter, mbsize=mbsize, ncheckins=ncheckins,
                with_rep=with_rep)



def scluster(matrix, centers, *,
             prior=0., msr=2, temp=1.,
             maxiter=1000, savepref="", weights=None, mbsize=-1):
    """
    This is a wrapper function that automatically calls the correct pyminicore scluster
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
        maxiter:
            Default 1000, maximum number of outer loops to iterate
        savepref:
            Default '\"\"'; If non-empty, then costs and assignments are mmap'd to disk
            to both reduce memory usage and store final cost/assignments for other analysis.

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
    if isinstance(matrix, sp.csr_matrix) or isinstance(matrix, csr_tuple):
        matrix = CSparseMatrix(matrix)
    import pyminicore as pmc
    return pmc.scluster(matrix, centers, msr=msr,
                        prior=prior, temp=temp, maxiter=maxiter,
                        savepref=savepref, weights=weights)


#  This function provides a single starting point for clustering start-to-finish
def cluster(data, *, msr, k, prior=0., seed=0, nmkc=0,
            ntimes=1, lspp=0, use_exponential_skips=False,
            n_local_trials=1, weights=None, mbsize=-1, clustereps=1e-4,
            temp=-1., cs=False, with_rep=True, outpref="mc.cluster.output",
            maxiter=50):
    """Convenience wrapper for hcluster and scluster which selects
        either hard or soft clustering
    """
    soft = temp > 0.  # Enable soft clustering by setting temperature
    if isinstance(data, csr_tuple):
        mcdata = CSparseMatrix(data)
    else:
        mcdata = data
    from pyminicore import kmeanspp as pykmpp
    ids, asn, costs = pykmpp(data, msr=msr, k=k,
                             prior=prior, seed=seed,
                             ntimes=ntimes, lspp=lspp, expskips=use_exponential_skips,
                             n_local_trials=n_local_trials, weights=weights)
    if soft:
        try:
            return scluster(mcdata, centers=ids, msr=msr, prior=prior, weights=weights, temp=temp, maxiter=maxiter,
                            savepref=outpref)
        except:
            raise NotImplementedError("Soft clustering not supported")
    else:
        return hcluster(data, centers=ids, prior=prior, msr=msr, weights=weights, eps=clustereps, maxiter=maxiter,
                        with_rep=with_rep, cs=cs)
