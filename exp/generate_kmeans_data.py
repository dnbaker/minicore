import numpy as np
import sys
np.random.seed(0)

num_clusters = 10
num_dim = 50
num_rows = 5000
assert num_rows % num_clusters == 0

centers = np.abs(np.random.standard_normal(size=(num_clusters, num_dim)) * 5.)

points = np.vstack([np.random.standard_normal(size=(num_rows // num_clusters, num_dim)) * 1 + centers[i,:][np.newaxis, :]
                    for i in range(num_clusters)])

ordering = np.arange(0, num_rows, dtype=np.uint32)
np.random.shuffle(ordering)
if sys.argv[1:]:
    ofp = open(sys.argv[1], "w")
    labels = sys.argv[1] + ".labels.txt"
else:
    ofp = open("random.out", "w")
    labels = "random.out.labels.txt"
ofp.write("%d/%d/%d\n" % (num_rows, num_dim, num_clusters))
for index in ordering:
    ofp.write(" ".join(map(str, points[index,:])) + "\n")
with open(labels, "w") as f:
    f.write("\n".join(str(ordering[i] // (num_rows // num_clusters)) for i in range(num_rows)))
    f.write("\n")

if ofp != sys.stdout: ofp.close()
