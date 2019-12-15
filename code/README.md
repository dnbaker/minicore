# Contents [![Build Status](https://travis-ci.com/dnbaker/fgc.svg?branch=master)](https://travis-ci.com/dnbaker/fgc)

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

Algorithms from:
```
Hu Ding, Haikuo Yu, Zixiu Wang  
Greedy Strategy Works for k-Center Clustering with Outliers and Coreset Construction                
```
1. kcenter with outiers, 2-approximation
2. kcenter bicriteria approximation
3. kcenter with outliers coreset construction (uses previous methods as a subroutine)

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
