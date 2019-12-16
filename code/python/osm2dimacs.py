"""

"""
import osmium as o
import sys
from math import cos, sin, asin, sqrt, pi as PI
from collections import Counter

EARTH_RADIUS_IN_METERS = 6372797.560856

def deg_to_rad(degree):
    return degree * (PI / 180.0);

def distance(c1, c2):
    lonh = sin(deg_to_rad(c1.lon - c2.lon) * 0.5);
    #lonh = sin(deg_to_rad(c1.x - c2.x) * 0.5);
    lonh *= lonh;
    lath = sin(deg_to_rad(c1.lat - c2.lat) * 0.5);
    # lath = sin(deg_to_rad(c1.y - c2.y) * 0.5);
    lath *= lath;
    tmp = cos(deg_to_rad(c1.lat)) * cos(deg_to_rad(c2.lat));
    # tmp = cos(deg_to_rad(c1.y)) * cos(deg_to_rad(c2.y));
    return 2.0 * EARTH_RADIUS_IN_METERS * asin(sqrt(lath + tmp * lonh));



class RoadLengthHandler(o.SimpleHandler):
    def __init__(self):
        super(RoadLengthHandler, self).__init__()
        self.length = 0.0
        self.edge_tups = []
        self.nodes = set()

    def way(self, w):
        if 'highway' in w.tags:
            try:
                # print(len(w.nodes), w.nodes)
                nn = len(w.nodes)
                for node in w.nodes:
                    self.nodes.add(node.ref)
                #fn = w.nodes[0]
                #print(dir(fn))
                # print("lat: %s. lon: %s. x: %s. y: %x" % (fn.lat, fn.lon, fn.x, fn.y))
                for pair in ([w.nodes[pi], w.nodes[pi + 1]] for pi in range(len(w.nodes) - 1)):
                    dist = distance(pair[0], pair[1])
                    self.edge_tups.append((pair[0].ref, pair[1].ref, distance(pair[0], pair[1])))
                    
                self.length += o.geom.haversine_distance(w.nodes)
            except o.InvalidLocationError:
                # A location error might occur if the osm file is an extract
                # where nodes of ways near the boundary are missing.
                print("WARNING: way %d incomplete. Ignoring." % w.id)

def print_header(fp, h):
    set(fp.write("c %s\n" % l) for l in [
        "Auto-generated 9th DIMACS Implementation Challenge: Shortest Paths-format file",
        "From Open Street Maps [OSM] (https://openstreetmap.org)",
        "Using pyosmium, a python wrapper for libosmium"
    ])
    fp.write("p sp %d %d\n" % (len(h.nodes), len(h.edge_tups)))
    fp.write("c %d nodes, %d edges\n" % (len(h.nodes), len(h.edge_tups)))
    fp.write("c \n")


def main(osmfile):
    h = RoadLengthHandler()
    # As we need the geometry, the node locations need to be cached. Therefore
    # set 'locations' to true.
    h.apply_file(osmfile, locations=True)

    print('Total way length: %.2f km' % (h.length/1000), file=sys.stderr)

    print('Num nodes: %d. Num edges: %d' % (len(h.nodes), len(h.edge_tups)), file=sys.stderr)
    # ng1 = [n for n, count in h.nodes.items() if count > 1]
    ofp = open(sys.argv[2], "w") if len(sys.argv) > 2 else sys.stdout
    print_header(ofp, h)
    nodeset = list(h.nodes)
    nodeset.sort()
    nodeid_d = {}
    for node_ind, node_ref in enumerate(nodeset):
        nodeid_d[node_ref] = node_ind
        ofp.write("c %d->%d\n" % (node_ref, node_ind))
    for lhs, rhs, dist in h.edge_tups:
        print("a %d %d %f" % (nodeid_d[lhs], nodeid_d[rhs], dist / 1000), file=ofp)
    #for tup in h.edge_tups:
    #     print(tup)

    return 0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python %s <osmfile>" % sys.argv[0])
        sys.exit(-1)

    exit(main(sys.argv[1]))

