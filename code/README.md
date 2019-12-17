# Contents [![Build Status](https://travis-ci.com/dnbaker/fgc.svg?token=nzWL3kpck4ymqu9SdesD&branch=master)](https://travis-ci.com/dnbaker/fgc)

1. [graph](#Graph)
    1. Wrappers for boost::graph
    2. Implementation of the Thorup 2005 algorithm
      1. Still missing the final algorithm for k-median, but this will likely soon be done with the local search with swaps method.
2. [coresets](#coreseth)
    1. `CoresetSampler` contains methods for building an importance sampling framework, performing sampling, and reweighting.
    2. IndexCoreset contains a vector of indices and a vector of weights.
    3. Methods for reducing are incomplete, but the software is general enough that this will not be particularly difficult.
        1. Each kind of coreset will likely need a different sort of merge/reduce, as our Coreset only has indices, not the data itself.
    4. [MatrixCoreset](#matrix_coreseth) creates a composable coreset managing its own memory from an IndexCoreset and a matrix.
3. [kcenter problem](#kcenterh)
4. [k-metric proble](#kmeansh)
5. [blaze-lib row/column iterator wrappers](#blaze_adaptorh)



## Graph

graph.h contains a wrapper for `boost::adjacency_list` tailored for k-median and other optimal transport problems.
bicriteria.h contains implementations from the Thorup 2005 paper on fast approximation of k-median on graphs.

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


## References

The k-center 2-approximation is [Gonzalez's](https://www.sciencedirect.com/science/article/pii/0304397585902245)
[algorithm](https://sci-hub.se/10.1016/0304-3975\(85\)90224-5).
The k-center clustering, 2-approximation, and coreset *with outliers* is [Ding, Yu, and Wang](https://arxiv.org/abs/1901.08219).

The importance sampling framework we use is from the [Braverman, Feldman, and Lang](https://arxiv.org/abs/1612.00889) paper from 2016,
while its application to graph metrics is from [Braverman, Huang, Jiang, Krauthgamer, and Wu](https://arxiv.org/abs/1907.04733).

We use the [Thorup](https://epubs.siam.org/doi/pdf/10.1137/S0097539701388884) paper [from 2005](https://sci-hub.se/10.1137/s0097539701388884)
for an initial graph bicriteria approximation.

### TODO

1. K-means
    1. Implement weighted k-means clustering
        1. Use Lloyd's algorithm
    2. Extensions for other metrics
2. Metric K-median
    1. Implement weighted k-median clustering on graph.
        a. Algorithm: http://homepage.divms.uiowa.edu/~kvaradar/sp2016/scribe\_week5.pdf
        b. Runtime: n^2+
    2. Jain/Navirazi
3. K-center
    1. Test accuracy of k-center method on real data
    2. Use coreset for clustering
4. Python bindings
    1. This will use the blaze::CustomMatrix interface to speak to Numpy and (maybe Torch?)
