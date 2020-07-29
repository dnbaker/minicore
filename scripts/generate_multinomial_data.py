import numpy as np
from numpy.random import poisson, multinomial

def maked(nd, basen): 
    import numpy as np 
    samples = poisson(4, size=(nd,)) 
    return poisson(basen), samples / np.sum(samples).astype(float) 

NGRP = 5

BASEN = 500

NDIM = 50

NPOINTS = 10000

dists = [maked(NDIM, BASEN) for i in range(NGRP)]

points_per_dist = np.array([poisson(1000) for i in range(NGRP)])
points_per_dist = np.ceil(points_per_dist / np.sum(points_per_dist) * NPOINTS)
diff = NPOINTS - np.sum(points_per_dist)
ppdr = range(len(points_per_dist))
if diff != 0:
    while diff < 0:
        p = np.random.choice(ppdr)
        if points_per_dist[p] > 0:
            points_per_dist[p] -= 1
            diff += 1
    while diff > 0:
        points_per_dist[np.random.choice(ppdr)] += 1
        diff -= 1

submats = [multinomial(myn, myp, size=(npoints,)) for
           npoints, (myn, myp) in zip(points_per_dist.astype(np.uint32), dists)]
points = np.vstack(submats)
for l in points:
    print("\t".join(map(str, l)))
with open("Summary.txt", "w") as f:
    ppd = points_per_dist.astype(np.uint32)
    ppdcs = np.cumsum(ppd)
    for np, cs in zip(ppd, ppdcs):
        f.write(f"{np}\t{cs}\n")
