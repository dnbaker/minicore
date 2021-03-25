## Minocore in Python

Much of the documentation is in doc-strings/type annotations.
Here, I am trying to provide more of a user guide.


## Installation

```bash
git clone --recursive https://github.com/dnbaker/minicore
cd minicore/python
# export CC if necessary to specify compiler
OMP_NUM_THREADS=8 python3 setup.py install
```


# Coreset generation
We have exposed a subset of functionality to Python. The `CoresetSampler` generates a coreset from a set of costs and a construction method,
while the more involved clustering code is also exposed.

# Clustering
For clustering Bregman divergences (squared distance, Itakura-Saito, and KL-divergence, for instance), kmeans++ sampling (via `kmeanspp`) provides accurate fast initial
centers, while `hcluster` performs EM from an initial set of points.

Since we're using the blaze linear algebra library, we need to create a sparse matrix for clustering from CSR format.


Example:
```python
import minicore mc
import numpy as np
import scipy.sparse as sp

data = # array of float32/float64s/int64/int32...
indices = # array of integers of 4 or 8 bytes
indptr = # array of integers of 4 or 8 bytes
shape = # np array or tuple

assert len(data) == len(indices)

csrtup = mc.csr_tuple(data=data, indices=indices, indptr=indptr, shape=shape, nnz=len(data))

# We could also produce one without copying:
csrmat = mc.CSparseMatrix(csrtup)

k = 50
beta = 0.5   # Pseudocount prior for Bregman divergences
             # smoothes some functions, and quantifies the "surprise" of having
             # unique features. The smaller, the more surprising.
             # See Witten, 2011

ntimes = 2   # Perform kmeans++ sampling %i times, use the best-scoring set of centers
             # defaults to 1

seed = 0     # if seed is not set, defaults to 0. Results will be identical with the same seed.

measure = "MKL" # See https://github.com/dnbaker/minicore/blob/main/docs/msr.md for examples/integer codes
                        # value can be integral or be the short string description
                        # MKL = reverse categorical KL divergence

weights = None  # Set weights to be a 1d numpy array containing weights of type (float32, float64, int, unsigned)
                # If none, unused (uniform)
                # otherwise, weights are used in both sampling and optimizing

centers, assignments, costs = mc.kmeanspp(csrmat, msr=measure, k=k, prior=beta, ntimes=ntimes,
                                          seed=seed, weights=weights)


lspprounds = 2 # Perform %i rounds of localsearch++. Yields a higher quality set of centers at the expense of more runtime

res = mc.hcluster(csrmat, centers, prior=beta, msr=measure,
                  weights=weights, lspp=lspprounds, seed=seed)


# res is a dictionary with the following keys:
#{"initcost": initial_cost, "finalcost": final_cost, "numiter": numiter,
# "centers": centers, # in paired sparse array format (data, idx), where idx is integral and data is floating-point
# "costs": costs  # cost for each point in the dataset
# "asn": assignments # integral, determining which center a point is assigned to.
#}
```

For measures that are not Bregman divergences (for which kmeans++ sampling may not work as well),
we can also use some discrete metric solvers for initial sets of points, but these are significantly slower.

We can also try greedy farthest-point sampling for initial centers. This is supported in the `minicore.greedy_select`, which uses a k-center approximation algorithm.

The options for this are governed by the minicore.SumOpts object, which holds basic metadata about a clustering problem.
If you set its `outlier_fraction` field to be nonzero, then this will use a randomized selection technique that is robust
to outliers and can also be used to generate a coreset, if the measure is a doubling metric.


## LSH

We also support a range of LSH functions.

JSD, S2JSDLSH, and the Hellinger LSH are hash functions for the JSD and its square root, and the Hellinger distance. This assumes that the rows have been normalized.
L1, L2, and PStable hashers should work regardless.



## Multithreading

By default, minicore uses the environmental variable `OMP\_NUM\_THREADS` number of threads.
This can be checked or changed within Python by accessing/modifying the `minicore.n_threads` object,
or via `get_num_threads` and `set_num_threads`.

Example:

```
import minicore as mc
mc.set_num_threads(40)  # Sets the number of threads to be 40
howmany = mc.get_num_threads()
assert howmany == 40
```

## Functions

1. kmeanspp -- kmeans++ sampling
2. hcluster -- hard clustering, with and without minibatch clustering. Set mbsize > 0 to enable minibatch clustering.
3. scluster -- soft clustering; Currently only supported with full (Lloyd's) iteration
4. SumOpts -- a set of options for clustering. Used as an entry point into greedy and d2 select.
5. minicore.greedy\_select -- greedy furthest points sampling. Set outlier\_fraction to be > 0 to allow outliers.
6. hvg -- Selects the most variable genes from a marix.
