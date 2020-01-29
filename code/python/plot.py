import matplotlib
import numpy as np
import sys
import matplotlib.pyplot as plt
matplotlib.use("Agg")

def print_items(*, data, xlabels, names, subgroup, prefix="default_out", end, filename):
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
    plt.xlabel("coreset size")
    plt.ylabel("distortion")
    plt.title("[%s] Coreset benchmark: %s" % (filename, subgroup))
    plt.plot()
    plt.savefig(f"{prefix}{subgroup}.png", dpi=300)
    plt.clf()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("infname")
    ap.add_argument("--prefix")
    ap.add_argument("--include-bfl", '-B', action='store_true')
    args = ap.parse_args()
    if args.prefix is None:
        args.prefix = args.infname
    filename = args.infname
    xlabels = []
    data = []
    for line in open(args.infname):
        if not line or line[0] == "#" or line[0] == '\n': continue
        xlabels.append(int(line.split()[0]))
        data.append(list(map(float, line.strip().split()[1:])))
    
    data = np.array(data)[2:]
    xlabels = np.array(xlabels)[2:]
    print(xlabels)
    # xlabels = np.ceil(np.log2(np.array(xlabels)[2:end])).astype(np.int32)
    #data = np.array(data)[2:end,:]
    # data = -np.log2(np.array(data))[2:end,:]
    # To make it clearer who was better
    mxfl, mxbfl, mxu = data[:,0], data[:,2], data[:,4]
    mufl, mubfl, muu = data[:,1], data[:,3], data[:,5]
    names = ("Varadarajan-Xiao", "BFL", "Uniform") if args.include_bfl else ("Varadarajan-Xiao", "Uniform")
    mudata=(mufl, mubfl, muu) if args.include_bfl else (mufl, muu)
    bflmudata = (mufl, mubfl, muu)
    if(data.shape[1] > 6):
        print("has extra fields")
        muffl, mufbfl, mufu = data[:,6], data[:,7], data[:,8]
        fdata = (muffl, mufbfl, mufu) if args.include_bfl else (muffl, mufu)
        bflfdata = (muffl, mufbfl, mufu)
    else:
        fdata = None
        mudata=(mufl, mubfl, muu) if args.include_bfl else (mufl, muu)
    mxdata= (mxfl, mxbfl, mxu) if args.include_bfl else (mxfl, mxu)
    bflmxdata = (mxfl, mxbfl, mxu)
    # data=(mufl, mubfl, muu)
    # names=("Varadarajan-Xiao", "BFL", "Uniform")
    #print_items(data=(mxfl, mxbfl, mxu), xlabels=xlabels, names=("Varadarajan-Xiao", "BFL", "Uniform"), subgroup="max", prefix=args.prefix)
    #print_items(data=(mxfl, mxbfl, mxu), xlabels=xlabels, names=("Varadarajan-Xiao", "BFL", "Uniform"), subgroup="max", prefix=args.prefix)
    for end in (12, 17):
        pref = args.prefix
        if args.include_bfl:
            pref += ".bfl"
        pref += ".%i" % end
        print("prefix: " + pref)
        print_items(data=mudata, xlabels=xlabels, names=names, subgroup="mean", prefix=pref, end=end, filename=filename)
        print_items(data=mxdata, xlabels=xlabels, names=names, subgroup="max", prefix=pref, end=end, filename=filename)
        if fdata is not None:
            print_items(data=fdata, xlabels=xlabels, names=names, subgroup="meanf", prefix=pref, end=end, filename=filename)
        '''
        if data.shape[1] > 6:
            lnames = ("Varadarajan-Xiao", "BFL", "Uniform")
            prefix = args.prefix + '.bfl.%i' % end
            print_items(data=bflmudata, xlabels=xlabels, names=lnames, subgroup="mean", prefix=pref, end=end)
            print_items(data=bflmxdata, xlabels=xlabels, names=lnames, subgroup="max", prefix=pref, end=end)
            if fdata is not None:
                print_items(data=bflfdata, xlabels=xlabels, names=lnames, subgroup="meanf", prefix=pref, end=end)
        '''
