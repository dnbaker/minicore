## Minicore: Fast Generic Coresets [![Build Status](https://travis-ci.com/dnbaker/minocore.svg?token=nzWL3kpck4ymqu9SdesD&branch=main)](https://travis-ci.com/dnbaker/minocore)

Minicore is a fast, generic library for constructing and clustering coresets on graphs, in metric spaces and under non-metric dissimilarity measures.
It includes methods for constant-factor and bicriteria approximation solutions, as well as coreset sampling algorithms.

These methods allow for fast and accurate summarization of and clustering of datasets with strong theoretical guarantees.

Minicore both stands for "mini" and "core", as it builds *concise representations* via *core*-sets, and as a portmanteau of Manticore and Minotaur.

![Minicore](https://raw.githubusercontent.com/dnbaker/minicore/ff4a0720758007c04400e0e7a87f585553670c6b/media/Tondo_Minotaur_London.processed.jpg "The Minicore")

### Dependencies

1. Boost, specifically the Boost Graph Library.
2. A compiler supporting C++17. We could remove this requirement without much work.
3. We conditionally use OpenMP. This is enabled with the setting of the OMP variable.

# Contents

1. [graph](#Graph)
    1. Wrappers for boost::graph
2. [coresets](#coreseth)
    1. `CoresetSampler` contains methods for building an importance sampling framework, performing sampling, and reweighting.
    2. IndexCoreset contains a vector of indices and a vector of weights.
    3. Methods for reducing are incomplete, but the software is general enough that this will not be particularly difficult.
        1. Each kind of coreset will likely need a different sort of merge/reduce, as our Coreset only has indices, not the data itself.
    4. [MatrixCoreset](#matrix_coreseth) creates a composable coreset managing its own memory from an IndexCoreset and a matrix.
3. Approximation Algorithms
    1. [k-center](#kcenterh) (with and without outliers)
    2. [k-means](#kmeansh)
    3. Metric k-median Problem
        1. Local search `lsearch.h`
        2. Jain-Vazirani `jv_solver.h`
    4. k,z-clustering using metric k-median solvers, exponentiating the distance matrix by z
    5. `[Thorup, 2005]`-sampling for pruning search space in both graph shortest-paths metrics and oracle metrics.
4. Optimization algorithms
    1. Expectation Maximization
        1. k-means
        2. Bregman divergences
        3. L1
            1. weighted median is complete, but it has not been retrofitted into an EM framework yet
5. [blaze-lib row/column iterator wrappers](#blaze_adaptorh)
    1. Utilities for working with blaze-lib
6. [disk-based matrix](#diskmath)
    1. Falls back to disk-backed data if above a specified size, uses RAM otherwise.
7. Streaming metric and `\alpha-`approximate metric clusterer
    1. `minicore/streaming.h`
8. Locality Sensitive Hashing
    1. LSH functions for:
        1. JSD
        2. S2JSD
        3. Hellinger
        4. L1 distance
        5. L2 distance
        6. L_p distance, 1 >= p >= 2
    2. LSH table
    3. TODO: multiprobe LSH tables
    4. See also [DCI](https://github.com/dnbaker/DCI) for an alternative view on LSH probing.


Soon, the goal is to, given a set of data, k, and a dissimilarity measure,
select the correct approximation algorithm, and sampling strategy to generate a coreset,
as well as optimize the clustering problem using that coreset.

For Bregman divergences and the squared L2 distance, D^2 sampling works.

For all other measures, we will either use the Thorup-sampled JV/local search approximation method
for metrics or the streaming k-means method for alpha-approximate metrics to achieve the approximate solution.

Once this is achieved and importances sampled, we optimize the problems:

1. EM
    1. Bregman, squared L2, Lp norms - Lloyd's
    2. L1 - Lloyd's, under median
    3. Log-likelihood ratio test, as weighted Lloyd's
2. General metrics
    1. Jain-Vazirani facility location solver
    2. Local search using swaps


There exist the potential to achieve higher accuracy clusterings using coresets compared with the full
data because of the potential to use exhaustive techniques. We have not yet explored this.


## Graph

graph.h contains a wrapper for `boost::adjacency_list` tailored for k-median and other optimal transport problems.

## kcenter.h

kcenter 2-approximation (farthest point)

Algorithm from:
```
T. F. Gonzalez. Clustering to minimize the maximum intercluster distance. Theoretical Computer Science, 38:293-306, 1985.
```

Algorithms from:
```
Hu Ding, Haikuo Yu, Zixiu Wang
Greedy Strategy Works for k-Center Clustering with Outliers and Coreset Construction
```
1. kcenter with outiers, 2-approximation
2. kcenter bicriteria approximation
3. kcenter with outliers coreset construction (uses Algorithm 2 as a subroutine)

## kmeans.h

1. k-means++ initialization scheme (for the purposes of an approximate solution for importance sampling)
2. k-means coreset construction using the above approximation
3. Weighted Lloyd's algorithm
4. KMC^2 algorithm (for sublinear time kmeans++)


## coreset.h

1. Importance-sampling based coreset construction
    1. Note: storage is external.
`IndexCoreset<IT, FT>`, where IT is index type (integral) and FT is weight type (floating point)

## matrix\_coreset.h

`MatrixCoreset<MatType, FT>` (Matrix Type, weight type (floating point)
Constructed from an IndexCoreset and a Matrix, simply concatenates both matrices and weight vectors.
Can be reduced using coreset construction.

## blaze\_adaptor.h

#### Wrappers
wrappers in the blz namespace for blaze::DynamicMatrix and blaze::CustomMatrix, with `rowiterator()` and `columniterator()`
functions allowing range-based loops over the the rows or columns of a matrix.

#### Norm structs
structs providing distances under given norms (effectively distance oracles), use in `kmeans.h`

#### diskmat.h
Uses [mio](https://github.com/mandreyel/mio) for mmap'd IO. Some of our software uses
in-memory matrices up to a certain size and then falls back to mmap.

### distance.h
Provides norms and distances.
Includes L1, L2, L3, L4, general p-norms, Bhattacharya, Matusita,
Multinomial and Poisson Bregman divergences, Multinomial Jensen-Shannon Divergence,
and the Multinomial Jensen-Shannon Metric, optionally with priors.

### applicator.h
Contains ProbDivApplicator, which is a caching applicator of a particular measure of dissimilarity.
Also contains code for generating D^2 samplers for approximate solutions.
Measures using logs or square roots cache these values.



## References

The k-center 2-approximation is [Gonzalez's](https://www.sciencedirect.com/science/article/pii/0304397585902245)
[algorithm](https://sci-hub.se/10.1016/0304-3975\(85\)90224-5).
The k-center clustering, 2-approximation, and coreset *with outliers* is [Ding, Yu, and Wang](https://arxiv.org/abs/1901.08219).

The importance sampling framework we use is from the [Braverman, Feldman, and Lang](https://arxiv.org/abs/1612.00889) paper from 2016,
while its application to graph metrics is from [Baker, Braverman, Huang, Jiang, Krauthgamer, and Wu](https://arxiv.org/abs/1907.04733v2).
We also support Varadarajan-Xiao, Feldman Langberg, and Bachem et al., methods for coreset sampling for differing dissimilarity measures.

We use a modified iterative version of the sampling from [Thorup](https://epubs.siam.org/doi/pdf/10.1137/S0097539701388884) paper from 2005
for an initial graph bicriteria approximation, which is described in the above Baker, et al. This can be found for shortest-paths graph metrics and oracle metrics in minicore/bicriteria.h.
