Framework
1. Rough estimate of solution.
2. Estimate sensitivity.
3. Sample points



### 1: Rough Estimate

**K-Center: The Classical Algorithm**
(Get 2-approximation)
```
A factor 2 approximation is classical [9, 13] and best possible [14], but the naturalalgorithm takes O(km) time. 
We get down to O(m) expected time, and our methodswill be reused for facility location.
```
0. All points nearest marked neighbor: See Thorup, 1.4. This requires a shortest path tree. This can be produced via Dijkstra.

```
With a distance oracle, we need O(n|S|) time to find all points nearest neighbor in S. However, when P is the vertex set of a graph and dist the shortest path distances, we can find all points nearest neighbor in S in near-linear time with a single source shortest path computation, no matter the size of S. More precisely, we have the following.
Observation 1. In a weighted connected graph with m edges, we can solve the all points nearest marked neighbor problem in O~(m) time.
QUICK k-MEDIAN AND FACILITY LOCATION FOR SPARSE GRAPHS 409
Proof. Introduce a source s with zero length edges to all a in S, and compute the shortest path tree to all vertices in O~(m) time. For each x, dist(x, S ) is the found distancefroms.
Also,foreachxinthesubtreeof a in S,wesetxS =a. Wenote that the same construction works for directed graphs if we first reverse the direction of all edges.
```
1. Binary search to detect the 'd' parameter for neighborhood size over Algorithm A, 
2. When it fits, then stick with this. (Linearithmic for it all.)

Cf.:

T. F. Gonzales,Clustering to minimize the maximum intercluster distance, Theoret. Comput.Sci., 38 (1985), pp. 293-550.
D. Hochbaum and D. B. Shmoys. A unified approach to approximation algorithms for bottle-neck problems, J. ACM, 33 (1986), pp. 533-550

**K-Median: The one we actually need**

My first attempt:
```
The core idea as I understand it is to get an approximate solution to k-median according to Thorup [QUICK k-MEDIAN, k-CENTER, AND FACILITY LOCATION FOR SPARSE GRAPHS], then use those weights to perform weighted sampling according to [Braverman, Feldman, Lang 2016], and then solving the full problem on that coreset.

So it seems that the core of the Thorup algorithm can be implemented as follows:
Build approximate distance oracle according to Thorup & Zwick [http://www.cs.jhu.edu/~baruch/teaching/600.427/Papers/oracle-STOC-try.pdf] with t = 2. Time: O(m * n^{1/2}). Space: n^{1.5}.
Use algorithms E and F in Thorup [https://epubs.siam.org/doi/pdf/10.1137/S0097539701388884] to sample points for a weighted set for calculating k-median solutions. [O(log^{2.5}(n)) time.]
Perform k-median solution on the subsetted points.

This should produce a (12 + o(1))-approximate solution to k-median over sparse graphs.

If I'm not mistaken, then sensitivities can be calculated using the formula 5.5 in [Braverman, Feldman, Lang], which are used to perform weighted sampling with replacement to create a coreset.

According to this paper, [Coresets for Clustering in Graphs of Bounded Treewidth: Braverman, Huang, Jiang, Krauthgamer, Wu], it is of size O(c·2^{2z}·kε^{−2}·(klog^{2}(k)·sdimmax+ log(1/δ))), where c is the approximation factor of the Thorup algorithm, and sdimmax is a bound on complexity.

As dimmax = O(tw(G)), this necessitates calculating tw(G), which we estimate by the cardinality of the set created by the union of B1 (the output of Balance-Decomp  [Alg 1]), and Boundary-Reduction(T, B1) [Alg 2], the output of the first algorithm.

And, lastly, there's the actual calculation of k-median results on the smaller weighted subsets, which I assume would be solved approximately using the typical Lloyd-style EM algorithm.

```

Shaofeng's response:
```
I think Mikkel’s 12-approx does not need to compute the distance oracle, and the final running time is O(kn), as in Proposition 12. Roughly, we run algorithm E first (which invokes algorithm D), and then use algorithm in [19] on the output of algorithm E.
The estimation of tw(G) is not from the algorithms Balance-Decomp  [Alg 1], and Boundary-Reduction(T, B1) [Alg 2] in  [Coresets for Clustering in Graphs of Bounded Treewidth: Braverman, Huang, Jiang, Krauthgamer, Wu]. In fact, these two algorithms are not part of our algorithm, and it is used only in the analysis.

To estimate tw(G), please have a look at this paper: https://arxiv.org/pdf/1901.06862.pdf. This paper gives treewidth estimation for many graph data sets, and you could try to use algorithms that are proposed in it (or algorithms that it cited).

We should of course estimate the treewidth of our data, but I think the coreset construction algorithm does not need to strictly sample so many points as in the theoretical bound. You could samples, say 0.1% - 1% of data, and evaluate the empirical accuracy (which should be good). In fact, the theoretical bound is far from tight in practice. For instance, the 1/eps^2 factor comes from the sampling bound of PAC learning, and as observed by many researchers, such a bound is never tight in practice. 

3.	You mentioned Lloyd-style EM algorithm. What is it exactly? In my mind, Lloyd algorithm only works in Euclidean spaces for k-means. In Euclidean space, in each iteration of Lloyd’s, we start with a set of k center points, and then compute the clustering w.r.t. those k points, and compute the new geometric means of the k clusters. The algorithm terminates if the new centers are close enough to the centers from last iteration.
However, the geometric mean is not defined in graphs, so the algorithm does not work; plus, our focus is k-median instead of k-means. One algorithm that should still work for k-median in graphs, is the 5-aprpox local search algorithm: http://theory.stanford.edu/~kamesh/lsearch.pdf. There might be other heuristics that works practically better than this local search, but unfortunately I’m not aware of any.
I understand that there are plenty of details in the implementation, so please feel free to ping me for discussion. 
```

So then this would mean:
Run algorithm E (using Dijkstra's). I can do this in SciPy after building the graph.

Is there a clear way to run this algorithm?

Let:

F = V
`n_c` = |V|
C = sparse matrix \in `\mathcal{R}^{n_c \times n_c}`, where `c_ij` = cost of connecting node i with node j
    `c_ij` is infinite unless otherwise specified
    this is also E
X = sparse matrix \in `\mathcal{R}^{n_c \times n_c}`, where `x_ij` = fractional relaxation of LP problem where node j is connected to facility i (node i, in our formulation)
Y = vector \in `\mathcal{R}^{n_c}`, where `y_i` = fractional relaxation of LP problem where `0 <= y_i <= 1` signifies that facility `i` is open, or that node i is a facility, in our formulation.

minimize:
np.matmul(C.reshape(1, -1).T, X.reshape(-1, 1)) + np.matmul([z z z \dots].T, Y)
subject to 
\hat{Y} > \hat{0}


#### 1.A: Solve clustering problem for Rough Estimate

### 2: Calculate Sensitivity

### 3: Sample
