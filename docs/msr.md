## Measure by Measure
| Measure | Category(ies) | Codename | Integer | Notes | 
|--------|-----------------|---------|----------|----|
| L1 distance | geometric metric | L1 | 0 | sum(\|x - \|y) |
| L2 distance | geometric metric | L2 | 1 | sqrt(sum((x - y)^2))|
| Total variation distance | geometric metric | TVD | 10 | Useful information theoretic measure; 1/2 L1 distance in probability space| 
| squared L2 distance | geometric rho-metric | SQRL2 | 2 | sum((x - y)^2)|
| Jensen-Shannon Metric | metric | JSM | 3 | Metric, but no closed-form solution for centroids. Perhaps best-suited for linkage/density-based clustering; sqrt(JSM)|
| Jensen-Shannon Divergence | pseudo-metric | JSD | 4 | Symmetric but no data-indepedent triangle inequality; convex combination of Bregman divergences|
| Itakura-Saito Divergence | Bregman divergence | IS | 16 | Corresponds to the exponential distribution; sum((x / y) - log(x / y) - 1)|
| Kullback-Leibler Divergence | Bregman divergence | MKL | 5 | Corresponds to the multinomial distribution with fixed n, or the categorical distribution; sum(x log(x / y))|
| Reversed Itakura-Saito Divergence | Bregman divergence | `REVERSE_ITAKURA_SAITO` | 17 | Corresponds to the multinomial distribution with fixed n, or the categorical distribution; sum(y / x) - log(y / x) - 1|
| Symmetric Itakura-Saito Divergence | Bregman divergence | SIS | 30 | Corresponds to the multinomial distribution with fixed n, or the categorical distribution|
| Hellinger Distance | metric | HELLINGER | 7 | Information-theoretic measure with useful upper/lower bounds for TVD and JSD; sqrt(sum((sqrt(x) - sqrt(y))^2))|
| Bhattacharyya Metric | metric | `BHATTACHARYYA_METRIC` | 8 | Non-metric, but useful measure of dissimilarity. Related to the metric.; sqrt(1 - sum(sqrt(x) * sqrt(y))) |
| Bhattacharyya Distance | dissimilarity measure | `BHATTACHARYYA_DISTANCE` | 9 | Non-metric, but useful measure of dissimilarity. Related to the metric;  -log(sum(sqrt(x) * sqrt(y)))| 
| LLR | dissimilarity measure | `LLR` | 11 | non-metric, but can be processed as a data-dependent rho-metric. As the prior rises, the $\rho$-violation of the triangle inequality shrinks; see below|
| UWLLR | dissimilarity measure | `UWLLR` | 14 | non-metric, but can be processed as a data-dependent rho-metric. As the prior rises, the $\rho$-violation of the triangle inequality shrinks. JSD and S2JSD LS hash functions will likely work for this and LLR; see below for formulation|


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

http://eduardovalle.com/wordpress/wp-content/uploads/2014/10/silva14sisapLargeScaleMetricLSH.pdf provides techniques for non-metrics; additionally, their use of kmeans++ for nearest neighbor lookup comparisons suggests that there may be a way to use importance sampling via D2 sampling as a hack around a proper LSH table.
