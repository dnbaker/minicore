import numpy as np
import sys
from argparse import ArgumentParser as agp
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components
from numpy import sqrt


def generate_line_graph(*, dim, npoints):
    linelen = 100.

    randproj = np.random.normal(size=(dim,))
    randproj /= np.linalg.norm(randproj)

    points_along_line = np.random.uniform(size=(npoints,), high=linelen)
    points = np.outer(points_along_line, randproj)
    assert points.shape[0] == npoints
    connections = [(i, i - 1) for i in range(1, npoints)]
    nrc = np.random.choice
    for i in range(npoints):
        connections += [(i, nrc(range(npoints))), (i, nrc(range(npoints)))]
    connections = (np.array([c[0] for c in connections]),
                   np.array([c[1] for c in connections]))
    dists = np.array([np.linalg.norm(points[i] - points[j])
                      for i, j in zip(*connections)])
    return connections, dists


def generate_random_graph(*, dim, npoints, norm):
    if not norm:
        norm = 1
    try:
        if int(norm) == norm:
            norm = int(norm)  # Turn 1. or 2. into integers
    except ValueError:
        pass
    points = np.random.standard_cauchy(size=(npoints, dim))
    connections = np.random.choice(npoints**2, replace=False,
                                   size=int(npoints * sqrt(np.log(npoints))))
    connections.sort()
    connections = ((connections / npoints).astype(np.int32),
                   connections.astype(np.int32) % npoints)
    cgraph = lil_matrix((npoints, npoints), dtype=np.bool)
    for lhs, rhs in zip(*connections):
        cgraph[lhs, rhs] = 1
    ccgraph = csr_matrix(cgraph)
    nc, labels = connected_components(csgraph=ccgraph,
                                      directed=False, return_labels=True)
    while nc != 1:
        print("Reconnecting because it wasn't connected: %d" % nc,
              file=sys.stderr)
        lhs = np.random.choice(npoints, size=20)
        rhs = np.random.choice(npoints, size=20)
        for l, r in zip(lhs, rhs):
            cgraph[l, r] = 1
        ccgraph = csr_matrix(cgraph)
        nc, labels = connected_components(csgraph=ccgraph,
                                          directed=False, return_labels=True)
    del ccgraph
    indices = cgraph.nonzero()
    dists = np.array([np.linalg.norm(points[i] - points[j], ord=norm)
                      for i, j in zip(*connections)])
    return connections, dists


a = agp()
aa = a.add_argument
aa("--dim", default=50, type=int)
aa("--npoints", default=500, type=int)
aa("--fmt", default="line")
aa("--norm", type=float)
a = a.parse_args()
dim, npoints = a.dim, a.npoints

if a.fmt == "line":
    connections, dists = generate_line_graph(dim=dim, npoints=npoints)
elif a.fmt == "random":
    connections, dists = generate_random_graph(dim=dim, npoints=npoints,
                                               norm=a.norm)
else:
    raise NotImplementedError("WHOOOOOOOOOOOOOOOOOOOOOOOOOOAAAAAAAAAAAAAAA")


print(f"c Auto-generated with {sys.argv[0]}, dim={dim}, npoints={npoints}")
print(f"p sp {npoints} {len(dists)}")
for d, start, end in zip(dists, connections[0], connections[1]):
    print(f"a {start + 1} {end + 1} {d}")
