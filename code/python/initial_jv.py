import numpy as np
import random

''' Be a bad person and use C++ keywords '''
true = True
false = False

np.random.seed(13)

nf = 5
nc = 25

city_locs = np.power(np.random.standard_cauchy((nc, 2)).astype(np.float64) + np.random.normal(size=(nc,2)), 2)
f_locs = np.power(np.random.standard_cauchy((nf, 2)).astype(np.float64) + np.random.normal(size=(nf,2)), 2)

c = np.array([[np.linalg.norm(f - c) for c in city_locs] for f in f_locs])
assert np.all(c > 0)

f_costs = np.array([1.] * nf)

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
istempopen = np.zeros(nf, dtype=np.bool)
isopen = np.zeros(nf, dtype=np.bool)

edges = [(c[i,j], i, j) for i in range(nf) for j in range(nc)]
edges.sort(key=lambda x: x[0])
edge_index = 0
assert all(edges[i][0] <= edges[i + 1][0] for i in range(nc * nf - 1))


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
    print("S [size: %u] : %s. nfac temp open: %d" % (len(S), S, np.sum(istempopen)))
    # Get minimum cost to make an edge go tight
    # We go through them in order, so it's easy
    mincostedge = edges[edge_index]        # edge of minimum cost
    minedgecost = mincostedge[0] - maxalph # how much to increment to make that edge go tight
    print("mincostedge cost: %f. alpha inc: %f (maxalph: %f)" % (mincostedge[0], minedgecost, maxalph))

    # Get minimum cost to open a facility
    minfaccost = np.inf
    minfacind = -1
    for fid in range(nf):
        if istempopen[fid]: continue
        cost_to_open = get_cost_to_open(w=w, c=c, fid=fid, v=v, f_costs=f_costs,s=S)
        assert cost_to_open >= 0.
        if cost_to_open < minfaccost:
            minfaccost = cost_to_open
            print("mfc: %f" % minfaccost)
            minfacind = fid
        # Cost = cost_fac - sum(willignness to pay for fid)
    if minedgecost < minfaccost:
        edge_index += 1  # Move to next edge index
        tighten_edge = true
    else:
        tighten_edge = false
    # print("minedgecost: %f. minfaccost: %f" % ((minedgecost, minfaccost)))
    inc = min(minedgecost, minfaccost)
    maxalph += inc
    # print("increment: %f" % inc)
    to_remove = set()
    if not tighten_edge:
        print("Sum of contributions before: %f" % np.sum(w[minfacind,:]))
    for s in S: 
        v[s] += inc
        assert v[s] == maxalph  # Meaning we don't necessarily have to update all the alphas until we remove them from the pool
        for fid in range(nf):
            if v[s] >= c[fid,s]:
                istight[fid,s] = true
                print("s is tight: %d" % s)
                if istempopen[fid]:
                    to_remove.add(s)
            # w[i,j] := max(0., v[j] - c[j][i]); in other words, w[j,i] is how much client i 
            oldw = w[fid,s]
            w[fid,s] = max(0., v[s] - c[fid,s])
            diff = w[fid,s] - oldw
            if diff > 0:
                print("contributions increased by %f" % diff)
            #print("for fid %d and s %d, v is %f and cost is %f. w[fid,s]: %f" % (fid, s, v[s], c[fid,s], w[fid,s]))
            #if fid == minfacind:
            #    print("sum for mfi: %f. Cost to open: %f" % (np.sum(w[minfacind,:]), f_costs[minfacind]))
    if not tighten_edge:
        assert minfacind >= 0
        print("minfacind: %d. cost to open: %f. sum of contributions: %f" % (minfacind, f_costs[minfacind], np.sum(w[minfacind,:])))
        # This means that we temporarily open a facility instead
        assert np.sum(f_costs[minfacind] <= np.sum(w[minfacind,:]))
        istempopen[minfacind] = true
        print("just opened fac %d" % minfacind)
        for cid in S:
            if v[cid] >= c[minfacind,cid]:  # If willingness to pay exceeds cost
                to_remove.add(cid)
        '''
        TODO:
        Handle if multiple open at exactly the same time
        for fid in range(nf):
            if np.sum(f_costs[minfacind] <= np.sum(w[minfacind,:])):
                istempopen
        '''

    for item in to_remove:  # Note: this way, we can ultimately only update them when we remove them
        v[item] = maxalph   # Which means that v updates may not be performed above as they are
    Sp += to_remove
    S -= to_remove

# Phase 2: electric boogaloo

tempopen = list(np.where(istempopen)[0])
random.shuffle(tempopen)
lastadded = tempopen[-1]

tprime = {lastadded}
while 1:
    contributors = np.where(w[lastadded,:] > 0.)[0]
    to_remove = set()
    for contrib in contributors:
        supporting = np.where(w[:,contrib] > 0.)[0]
        to_remove.update(supporting)
    tempopen = list(set(tempopen) - to_remove)
    if not tempopen: break
    lastadded = random.choice(tempopen)

t = tprime
print("Set: %s" % t)



    #print("NOT FINISHED")
    #break
