import minicore as mc
import scipy.sparse as sp
import numpy as np
import os
import time
import argparse as agp


measures_to_use = [2, 5, 12, 7, 4, 11]

beta_set = [0., .1, 1]

KSET = {5, 15, 25, 50, 250, 500}

beta_vals = {5: beta_set}


ap = agp.ArgumentParser()
ap.add_argument("--trials", default=3, help="Number of trials to use. must be odd (for median)", type=int)
ap.add_argument("--num-threads", '-p', default=1, help="Number of threadsto use.", type=int)
ap.add_argument("--outdir", default="outdir", help="base output directory")
ap.add_argument("--mbsize", default=-1, help="minibatch size. If unset or less than 0, performs full EM.", type=int)


args = ap.parse_args()

from load_files import all_experiments, ordering
ntrials = args.trials
num_threads = args.num_threads
mbsize = args.mbsize
outdir = args.outdir
mc.set_num_threads(num_threads)


def run_experiment(resultdir, msr, matrix, k, beta):
    times = [0] * ntrials
    results = [[] for i in range(ntrials)]
    for i in range(ntrials):
        start = time.time()
        results[i] = mc.kmeanspp(matrix, msr=msr, betaprior=beta, k=k)
        stop = time.time()
        times[i] = start - stop
    argmed = times.index(np.median(times))
    return (times[argmed], results[argmed], k, beta)


def compute_experiment(resultdir, msr, matrix, kset, betavalset):
    ret = []
    for k in kset:
        for beta in betavalset:
            ret.append(run_experiment(resultdir, msr, matrix, k, beta))
            print("for path " + resultdir + ", median time for %d %f is %f" % (k, beta, ret[-1][0]))


for exp_name in ordering:
    matrix = all_experiments[exp_name]
    print("About to parse matrix with name %s" % exp_name, file=sys.stderr)
    mcmat = mc.smw(matrix)
    for msr in measures_to_use:
        startt = time.time()
        betavalset = beta_vals.get(msr, [0.])
        msrname = mc.meas2str(msr)
        resultdir = "/".join([outdir, exp_name, msrname])
        os.make_dirs(resultdir)
        results = compute_experiment(resultdir, msr, mcmat, KSET, betavalset)
        with open(resultdir + "/timing.txt", "w") as f:
            print("\t".join(["Experiment", "Measure", "K", "Prior", "Total cost"]), file=f)
            for median_time, result, k, beta in results:
                total_cost = str(np.sum(result[2]))
                print("\t".join([exp_name, msrname, str(k), str(beta), total_cost]), file=f)
        for _, result, k, beta in results:
            result[0].tofile(resultdir + "/centers.%s.%d.%f.npy" % (result[0].dtype.char, k, beta))
            result[1].tofile(resultdir + "/asn.%s.%d.%f.npy" % (result[1].dtype.char, k, beta))
            result[2].tofile(resultdir + "/costs.%s.%d.%f.npy" % (result[2].dtype.char, k, beta))
        stopt = time.time()
        print(f"Computed kmeans++ timings for dataset {exp_name} for measure {msrname} in {stopt - startt}s")
