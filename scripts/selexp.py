import sys
import math
import uuid
import subprocess
from subprocess import check_call


DISTS = {
    'L1': '1', 'L2': '2', 'SQRL2': 'S', 'MKL': 'M',
    'JSM': 'J', 'JSD': 'j', 'PSL2': 'P',
    'PL2': 'Q', 'TVD': 'T', 'HELLINGER': 'H',
    'BHATTACHARYYA_DISTANCE': 'Y', 'BHATTACHARYYA_METRIC': 'b',
    'COSINE_DISTANCE': 'U',
    'PROBABILITY_COSINE_DISTANCE': 'u',
    'REVERSE_MKL': 'R',
    'ITAKURA_SAITO': 'i'
}

MEASURES = list(DISTS.keys())

# [Greedy, D2, D2 with one round, cluster]
CMDZIP = ["greedy", "d2", "d2p1", "cluster"]
CMDDICT = {k: [True, True, False, False] for k in DISTS.keys()}
PRIORDICT = {k: False for k in DISTS.keys()}
DEFAULT_SAMPLE_TRIES = 10
VANILLA_METRICS =  ('TVD', 'L1', 'L2', 'PL2', 'PSL2', 'SQRL2')
for k in VANILLA_METRICS:
    CMDDICT[k][2] = CMDDICT[k][3] = True

for k in ("JSM", "JSD", "HELLINGER", "MKL", "REVERSE_MKL", "ITAKURA_SAITO"):
    PRIORDICT[k] = [1e-4, 1e-2, 1]


PRIORS = {"DIRICHLET", "NONE", "GAMMA_BETA"}
COMMAND = {"GREEDY": " -G ", "D2": " -l ", "CLUSTER": "", "DOUBLINGCS": " -7 "}
SENSES = {"FL": " -F ", "VX": " -V ", "BFL": "", "LBK": ' -E '}


class Run:
    def __init__(self, k=10, *, cmd, key, gammabeta=1., sample_tries=DEFAULT_SAMPLE_TRIES, lloyd_iter=100, seed=1337, sm="BFL", threads=None, dest=None, path=None, prior="NONE", r=None, of=0., timefunc=None, pb=False):
        assert key in DISTS, "key must be in %s" % (','.join(DISTS.keys()))
        assert prior in PRIORS, f"{prior} not in {','.join(PRIORS)}"
        assert cmd in COMMAND, f"{cmd} not in {','.join(COMMAND)}"
        assert sm in SENSES, f"{sm} not in {','.join(SENSES)}"
        self.k = k
        self.key = key
        self.path = path
        self.dest = dest
        self.prior = prior
        self.sample_tries = sample_tries
        self.lloyd_iter = lloyd_iter
        self.cmd = cmd
        self.seed = seed
        self.sm = sm
        self.gammabeta = gammabeta
        if self.prior == 'DIRICHLET':
            self.gammabeta = 1.
        self.threads = threads
        self.r = r
        self.of = of
        self.timefunc = timefunc
        self.pb = pb


    # Pass a range to r (list, etc.) to perform multiple times.

    def call(self, path=None, dest=None, r=None, seed=0):
        from subprocess import Popen, PIPE
        if not path:
            path = self.path
            if not path: raise Exception("")
        if not dest:
            dest = self.dest
            if not dest:
                dest = path
        if not r and self.r:
            r = self.r
        if r:
            return list(map(
                lambda x: self.call(path, f"{dest}.{x}", seed=seed+x, r=None), r))
        if not seed:
            seed = self.seed
        if not dest:
            dest = self.dest
            if not dest:
                dest = path
        cmd = f"mtx2cs {COMMAND[self.cmd]} -{DISTS[self.key]} {SENSES[self.sm]}"
        cmd += f" -L{self.lloyd_iter} -k{self.k} "
        if self.prior != 'NONE':
            cmd = cmd + f" -g{self.gammabeta} "
        if self.threads:
            cmd += f" -p{int(self.threads)} "
        cmd += f" -s{seed} "
        if self.of and self.cmd == "GREEDY":
            assert 1. >= self.of  >= 0., f"{self.of} out of range"
            cmd = cmd + f" -O{of} "
        pbs = " -B " if self.pb else ""
        if self.pb:
            print("parsing blaze")
        cmd = f"{cmd} {pbs} {path} {dest} &> {uuid.uuid1()}.log"
        if self.pb and " -B " not in cmd:
            raise Exception("S")
        if self.timefunc:
            cmd = f"{self.timefunc} -v {cmd}"
        #print("About to call '%s'" % cmd)
        #print("dest", dest, "path", path)
        try:
            check_call(cmd, shell=True)
        except subprocess.CalledProcessError as e:
            print("Failed to call '%s', continuing" % cmd, file=sys.stderr)



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=2000)
    ap.add_argument("-s", "--seed", type=int, default=0)
    ap.add_argument("path")
    ap.add_argument("--times", "-t", type=int, default=0)
    ap.add_argument("--dest", default=None)
    ap.add_argument("--lloyd-iter", type=int, default=100)
    ap.add_argument("--logfile", "-l", type=str)
    ap.add_argument("--threads", "-p", type=int, default=1)
    ap.add_argument("--outlier-fraction", "-O", type=float, default=0.)
    ap.add_argument("--timefunc", '-T', type=str, default=None)
    ap.add_argument("--parse-blaze", '-B', action='store_true')
    ap.add_argument("--skipearly", action='store_true')
    args = ap.parse_args()
    k = args.k
    path = args.path
    dest = args.dest if args.dest else path
    li = args.lloyd_iter
    ret = []
    rng = None if not args.times else range(args.times)
    lf = args.logfile
    threads = args.threads
    of = args.outlier_fraction
    tf = args.timefunc
    if not lf:
        if dest:
            lf = dest
        else:
            from functools import reduce
            lf = reduce(lambda x, y: x ^ hash(y), sys.argv, 0)
            lf = f"logfile{reduce(lambda x, y: x ^ hash(y), sys.argv, 0)}"
            print("No logfile provided, using '%s'" % lf, file=sys.stderr)
    pb = args.parse_blaze
    if args.skipearly:
        msr = [i for i in MEASURES if i not in {"L1", "L2", "SQRL2", "MKL"}]
    else:
        msr = MEASURES[:]
    for m in msr:
        print(f"processing {m}", file=sys.stderr)
        cmds = CMDDICT[m]
        mydest = dest + "_" + m
        # print("dest is %s and mydest is %s" % (dest, mydest))
        s = PRIORDICT[m]
        def perform_run(with_cmd, md=mydest, lf=of):
            ret = []
            if s:
                for g in s:
                    runner = Run(cmd=with_cmd, key=m, k=k, path=path, dest=md, gammabeta=g, seed=args.seed, prior="GAMMA_BETA", lloyd_iter=li, threads=threads, of=lf, timefunc=tf, pb=pb)
                    print("Callign runners")
                    runner.call(path=path, dest=md, r=rng)
            else:
                runner = Run(cmd=with_cmd, key=m, k=k, seed=args.seed, lloyd_iter=li, threads=threads, of=lf, timefunc=tf, pb=pb)
                runner.call(path=path, dest=md)
        if cmds[0]:
            print("Doing Greedy", file=sys.stderr)
            if of:
                perform_run("GREEDY", md=mydest + "_GREEDY_OUTLIERS_%f_" % of, lf=of)
            perform_run("GREEDY", md=mydest + "_GREEDY", lf=0.)
        if cmds[1]:
            print("Doing D2", file=sys.stderr)
            perform_run("D2", md=mydest + "_D2")
        if cmds[1]:
            print("Doing DOUBLINGCS", file=sys.stderr)
            perform_run("DOUBLINGCS", md=mydest + "_DOUBLINGCS")
        if cmds[3]:
            print("Doing CLUSTER", file=sys.stderr)
            perform_run("CLUSTER", md=mydest + "_CLUSTER")
        '''
        if cmds[2]:
            tmpli = li
            li = 1
            ret += perform_run("CLUSTER", md=mydest + "_CLUSTER_1_ITER")
            li = tmpli
        '''
    sys.exit(0)


__all__ = [
    "Run", "DISTS", "MEASURES", "CMDZIP", "PRIORS", "PRIORDICT", "COMMAND", "VANILLA_METRICS", "CMDDICT", "SENSES"
]
