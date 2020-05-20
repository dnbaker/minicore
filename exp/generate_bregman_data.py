import numpy as np
import sys
import argparse

try:
    from cytoolz import frequencies as Counter
except ImportError:
    from collections import Counter
np.random.seed(0)

ap = argparse.ArgumentParser()
ap.add_argument("--num-clusters", type=int, help="Number of clusters.", default=10)
ap.add_argument("--num-rows", type=int, help="Number of rows.", default=5000)
ap.add_argument("--num-dim", type=int, help="Number of dimensions.", default=50)
ap.add_argument("--set-noise", type=float, default=1.)
ap.add_argument("--set-data-variance", type=float, default=5.)
ap.add_argument("--outfile", type=str, default="randombregman.out")
ap.add_argument("--sample-coverage", type=int, default=1000)
ap = ap.parse_args()

num_clusters = ap.num_clusters
num_dim = ap.num_dim
num_rows = ap.num_rows

assert num_rows % num_clusters == 0, "num rows must be divisible by number of clusters"

# Normalize
centers = np.abs(np.random.standard_cauchy(size=(num_clusters, num_dim)) * ap.set_data_variance)

centers = (1. / np.sum(centers, axis=1))[:,np.newaxis] * centers

datapoints = []
for i in range(num_clusters):
    for j in range(num_rows // num_clusters):
        # Generate a number of samples, and then sample them.
        nsamp = np.random.poisson(ap.sample_coverage)
        row = centers[i] + np.random.standard_normal(size=(num_dim,))
        row = np.abs(row)
        row /= np.sum(row)
        selections = Counter(np.random.choice(len(row), p=row, size=(nsamp,))[:])
        samples = np.zeros((num_dim,))
        for k, v in selections.items():
            samples[k] = v
        datapoints.append(samples)

datapoints = np.vstack(datapoints)

ordering = np.arange(0, num_rows, dtype=np.uint32)
np.random.shuffle(ordering)
with open(ap.outfile, "w") as ofp:
    ofp.write("%d/%d/%d\n" % (num_rows, num_dim, num_clusters))
    for index in ordering:
        ofp.write(" ".join(map(str, datapoints[index,:])) + "\n")
with open(ap.outfile + ".labels.txt", "w") as f:
    f.write("\n".join(str(ordering[i] // (num_rows // num_clusters)) for i in range(num_rows)))
    f.write("\n")
