from numpy import random, ascontiguousarray
import minicore as mc
from sklearn.datasets import make_blobs
import numpy as np

dat = np.random.rand(10000).reshape(1000, 10)

res = mc.kmeanspp(dat, k=25, msr=2)
out, asn, costs  = res

cs = mc.CoresetSampler()

sensid = mc.constants.SENSDICT["LBK"] # This uses coresets for Bregman divergences

cs.make_sampler(25, costs=costs, assignments=asn, sens=4)

weights, ids = cs.sample(100)
top = sorted(zip(ids, weights), key=lambda x: -x[1])[:10]
print("top 10: ", top)
