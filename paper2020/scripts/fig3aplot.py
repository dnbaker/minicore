#!/usr/bin/env python3
import matplotlib
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
matplotlib.use("Agg")

def print_items(*, data, xlabels, names, subgroup, prefix="default_out", end, filename, title=None):
    xlabels = xlabels[:end]
    data = tuple(d[:end] for d in data)
    #fig, ax = plt.subplots()
    #lines = [ax.plot(xlabels, f)[0] for f in data]
    for f in data:
        plt.plot(xlabels, f)
    #print(lines)
    plt.legend(names)
    #legends = [ax.legend(handles=[line], label=name) for line, name in zip(lines, names)]
    # *zip(lines, names))
    plt.xlabel("Coreset Size")
    plt.ylabel("Empirical Error")
    if title is None:
        plt.title("[%s] Coreset accuracy benchmark: %s" % (filename, subgroup))
    else:
        plt.title(title)
    plt.plot()
    plt.savefig(f"{prefix}{subgroup}.png", dpi=300)
    plt.clf()


def path2dat(path):
    xlabels = []
    data = []
    for line in open(path):
        if not line or line[0] == "#" or line[0] == '\n': continue
        xlabels.append(int(line.split()[0]))
        data.append(list(map(float, line.strip().split()[1:])))

    data = np.array(data)[2:]
    xlabels = np.array(xlabels)[2:]
    return xlabels, data

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("infnamefile")
    ap.add_argument("counts")
    #ap.add_argument("names")
    ap.add_argument("--include-bfl", '-B', action='store_true')
    args = ap.parse_args()
    filename = args.infnamefile
    paths = list(map(str.strip, open(filename)))
    for path in paths:
        assert os.path.isfile(path), path + 'is not a file'
    dats = list(map(path2dat, paths))
    #names = list(map(str.strip, open(args.names)))
    #assert len(names) == len(dats)
    xlabels = dats[0][0]
    counts = [l.strip() for l in open(args.counts)]
    assert len(dats) == len(counts)
    assert all(np.all(dats[0][0] == dats[i][0]) for i in range(1, len(paths)))
    maxes = np.vstack([d[:,0] for _, d in dats])
    names = list(map(str, counts))
    print(maxes.shape)
    print(xlabels)
    #print(maxes)
    #print("\n".join(", ".join(map(str, maxes[i,:])) for i in range(maxes.shape[0])))
    plt.plot(xlabels, maxes.T)
    plt.xlabel("Coreset Size", fontsize=16)
    plt.ylabel("Empirical Error", fontsize=16)
    plt.legend(names)
    plt.savefig(args.infnamefile + ".save.svg")
    plt.savefig(args.infnamefile + ".save.png", dpi=600)
    for n in (9, 13, 17, 22):
        plt.clf()
        #print(maxes.shape)
        subx = xlabels[:n]
        submax = maxes.T[:n,:]
        for subset, ls in zip(submax.T, ['-', '-.', ':', '--']):
            plt.plot(subx, subset, linestyle=ls)
        plt.xlabel("Coreset Size", fontsize=16)
        plt.ylabel("Empirical Error", fontsize=16)
        plt.legend(names)
        plt.savefig(args.infnamefile + "%d.save.svg" % n)
        plt.savefig(args.infnamefile + "%d.save.png" % n, dpi=600)

    sys.exit(0)
