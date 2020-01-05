import numpy as np
import random

true = True
false = False

np.random.seed(13)

nf = 15
nc = 1000

city_locs = np.power(np.random.standard_cauchy((nc, 2)).astype(np.float64) + np.random.normal(size=(nc,2)), 2)
f_locs = np.power(np.random.standard_cauchy((nf, 2)).astype(np.float64) + np.random.normal(size=(nf,2)), 2)

c = np.array([[np.linalg.norm(f - c) for c in city_locs] for f in f_locs])
assert np.all(c > 0)

f_costs = np.array([10.] * nf)


### TODO: 1. measure the actual cost of the solution.
###       2. Plug this into C++ code base.
###       3. Get coresets.
###       4. Sanity check: naive graphs (straight line, maybe circles)
###


print("costs:", c.astype(np.int32))
print("costs to open facilities: ", f_costs.astype(np.int32))

# At this point, the problem statement is c (cost matrix) and f (cost of opening facilities)
# We generate using Euclidean distance just to ensure that it follows 

v = np.zeros(nc)
w = np.zeros((nf, nc))
assert w.shape == c.shape

S = set(range(nc))
Sp = []

istight = np.zeros((nf, nc), dtype=np.bool)
tos = set()
ntos = set(range(nf))
isopen = np.zeros(nf, dtype=np.bool)

edges = [(c[i,j], i, j) for i in range(nf) for j in range(nc)]
edges.sort(key=lambda x: -x[0])
etop = lambda: edges[-1]
epop = lambda: edges.pop()
assert all(edges[i][0] >= edges[i + 1][0] for i in range(nc * nf - 1))


def get_cost_to_open(*, w, c, fid, v, f_costs,s):
    ntight = np.sum(istight[fid,list(s)])
    diff = f_costs[fid] - np.sum(w[fid,:])
    ret = diff / ntight if ntight else np.inf
    #print(f"diff: {diff}. ntight: {ntight}. ret: {ret}")
    #print("ret: %f" % ret)
    return ret



maxalph = 0.



# Phase 1
while S:
    print("S [size: %u]. nfac temp open: %d" % (len(S), len(tos)))
    # Get minimum cost to make an edge go tight
    # We go through them in order, so it's easy
    mincostedge = etop()        # edge of minimum cost
    minedgecost = mincostedge[0] - maxalph # how much to increment to make that edge go tight
    #print("mincostedge cost: %f. alpha inc: %f (maxalph: %f)" % (mincostedge[0], minedgecost, maxalph))

    # Get minimum cost to open a facility
    minfaccost = np.inf
    minfacind = -1
    for fid in ntos:
        cost_to_open = get_cost_to_open(w=w, c=c, fid=fid, v=v, f_costs=f_costs,s=S)
        assert cost_to_open >= 0.
        if cost_to_open < minfaccost:
            minfaccost = cost_to_open
            #print("mfc: %f" % minfaccost)
            minfacind = fid
        # Cost = cost_fac - sum(willignness to pay for fid)
    if minedgecost < minfaccost:
        epop()  # Move to next edge index
        tighten_edge = true
    else:
        tighten_edge = false
    # print("minedgecost: %f. minfaccost: %f" % ((minedgecost, minfaccost)))
    inc = min(minedgecost, minfaccost)
    maxalph += inc
    # print("increment: %f" % inc)
    to_remove = set()
    #if not tighten_edge:
    #    print("Sum of contributions before: %f" % np.sum(w[minfacind,:]))
    for s in S: 
        v[s] += inc
        assert v[s] == maxalph  # Meaning we don't necessarily have to update all the alphas until we remove them from the pool
        for fid in range(nf):
            if v[s] >= c[fid,s]:
                istight[fid,s] = true
                #print("s is tight: %d" % s)
                if fid in tos:
                    to_remove.add(s)
            # w[i,j] := max(0., v[j] - c[j][i]); in other words, w[j,i] is how much client i 
            oldw = w[fid,s]
            w[fid,s] = max(0., maxalph - c[fid,s])
            diff = w[fid,s] - oldw
            if diff > 0:
                print("contributions increased by %f" % diff)
    if not tighten_edge:
        assert minfacind >= 0
        print("minfacind: %d. cost to open: %f. sum of contributions: %f" % (minfacind, f_costs[minfacind], np.sum(w[minfacind,:])))
        # This means that we temporarily open a facility instead
        contrib_sum = np.sum(w[minfacind,:])
        fc = f_costs[minfacind]
        assert fc <= contrib_sum + 1e-10, f"diff: {abs(f_costs[minfacind] - np.sum(w[minfacind,:]))}"
        tos.add(minfacind)
        ntos.remove(minfacind)
        print("just opened fac %d" % minfacind)
        for cid in S:
            if v[cid] >= c[minfacind,cid]:  # If willingness to pay exceeds cost
                to_remove.add(cid)
        '''
        TODO:
        Handle if multiple open at exactly the same time
        for fid in range(nf):
            if np.sum(f_costs[minfacind] <= np.sum(w[minfacind,:])):
        '''

    for item in to_remove:  # Note: this way, we can ultimately only update them when we remove them
        v[item] = maxalph   # Which means that v updates may not be performed above as they are
    Sp += to_remove
    S -= to_remove

# Phase 2: electric boogaloo

tempopen = list(tos)
for fac in tempopen:
    assert sum(w[fac,:] >= 0.)
random.shuffle(tempopen)
lastadded = tempopen.pop()

tprime = {lastadded}
while 1:
    print("tempopen: %s" % tempopen)
    contributors = np.where(w[lastadded,:] > 0.)[0]
    to_remove = set()
    for contrib in contributors:
        supporting = np.where(w[:,contrib] > 0.)[0]
        to_remove.update(supporting)
    print("removing: %s" % to_remove)
    if to_remove:
        tempopen = list(set(tempopen) - to_remove)
    if not tempopen: break
    random.shuffle(tempopen)
    lastadded = tempopen.pop()
    tprime.add(lastadded)

t = tprime
print("Set: %s" % t)



    #print("NOT FINISHED")
    #break
