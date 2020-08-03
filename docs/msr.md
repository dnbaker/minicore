## Measure by Measure
| Measure | Category(ies) | Codename | Integer | Notes | 
|--------|-----------------|---------|----------|----|
| L1 distance | geometric metric | L1 | 0 | |
| L2 distance | geometric metric | L2 | 1 | |
| Total variation distance | geometric metric | TVD | 10 | Useful information theoretic measure; 1/2 L1 distance in probability space | 
| squared L2 distance | geometric rho-metric | SQRL2 | 2 | |
| Jensen-Shannon Metric | metric | JSM | 3 | Metric, but no closed-form solution for centroids. Perhaps best-suited for linkage/density-based clustering|
| Jensen-Shannon Divergence | pseudo-metric | JSD | 4 | Symmetric but no data-indepedent triangle inequality; convex combination of Bregman divergences|
| Itakura-Saito Divergence | Bregman divergence | IS | 16 | Corresponds to the exponential distribution|
| Kullback-Leibler Divergence | Bregman divergence | MKL | 5 | Corresponds to the multinomial distribution with fixed n, or the categorical distribution|
| Reversed Itakura-Saito Divergence | Bregman divergence | `REVERSE_ITAKURA_SAITO` | 17 | Corresponds to the multinomial distribution with fixed n, or the categorical distribution|
| Symmetric Itakura-Saito Divergence | Bregman divergence | SIS | 30 | Corresponds to the multinomial distribution with fixed n, or the categorical distribution|
| Hellinger Distance | metric | HELLINGER | 7 | Information-theoretic measure with useful upper/lower bounds for TVD and JSD|
| Bhattacharyya Metric | metric | `BHATTACHARYYA_METRIC` | 8 | Non-metric, but useful measure of dissimilarity. Related to the metric. | 
| Bhattacharyya Distance | dissimilarity measure | `BHATTACHARYYA_DISTANCE` | 9 | Non-metric, but useful measure of dissimilarity. Related to the metric. | 
| LLR | dissimilarity measure | `LLR` | 11 | non-metric, but can be processed as a data-dependent rho-metric. As the prior rises, the $\rho$-violation of the triangle inequality shrinks |
| UWLLR | dissimilarity measure | `UWLLR` | 14 | non-metric, but can be processed as a data-dependent rho-metric. As the prior rises, the $\rho$-violation of the triangle inequality shrinks. JSD and S2JSD LS hash functions will likely work for this and LLR|


### Metrics, pseudometrics, and measures
Of the above, there are strict metrics: L1, L2, JSM, Hellinger, Bhattacharyya, and TVD

Some are constant-factor $\rho$-metrics (SQRL2)

Some are unbounded in theory but practically well-behaved

SIS, IS, KL, JSD, JSM, UWLLR, LLR


### Bregman Divergences
Jensen-Shannon, Itakura-Saito, and KL divergence are all Bregman divergences (and convex combinations thereof)
Itakura-Saito corresponds to the exponential distribution, while Jensen-Shannon corresponds to the multinomial.
Jensen-Shannon is the symmetrized KL divergence; the corresponding symmetrization for Itakura-Saito is available as SIS.

Squared L2 distance itself is also a Bregman divergence; it corresponds to the spherical Gaussian.


### LSH functions
L1 nearest-neighbors can be found with Cauchy sampling and p-stable distributions.
L2 uses Gaussian sampling + p-stable distributions.

Hellinger, JSD, LLR, and UWLLR nearest neighbors can be found using Hellinger/JSD LSH function, which is itself a reduction to the L2 nearest neighbors for the square roots of the probability vectors.
Because of the relations between TVD, JSD, and Hellinger, techniques that work for one often work for another.
