__author__ = 'sean'

import numpy as np
import matplotlib.pylab as plt
import star
import simple
import pickle
import time

def pspace():
    # heatmap: return np.arange(.1, 1., .1)
    # brief:
    return np.arange(.1, 1., .4)
    # briefer:
    # return np.arange(.1, 1., .8)
    # return np.arange(.8, 1., .1)

def nspace():
    return np.array([int(np.floor(x)) for x in np.exp(np.arange(.6,5,.3))])
    # brief:
    # return np.arange(10, 100, 80)

def nxspace():
    return np.array([int(np.floor(x/10)*10) for x in np.exp(np.arange(2.9,7.2,.3))])
    # brief:
    # return np.arange(10, 100, 80)

def nreps():
    return 4
    # brief:
    # return 2

def mRepetitions():
    return 500000
    # brief:
    # return 10000

def kRepetitions():
    return 10000
    # brief:
    # return 1000

def rDelta():
    return 1000

def confidence_level():
    return 0.95

def OutFileName():
    return 'out2_run.pkl'

def XOutFileName():
    return 'out2_run_x.pkl'

def InFileName():
    return 'in2_run.pkl'

def XInFileName():
    return 'in2_run_x.pkl'

def OldInFileName():
    return 'in_run.pkl'

def compute_alt(B, Balt):
    S = star.Star(B,Balt,mReps=mRepetitions())
    print "Rooting Table"
    S.root_table(k=kRepetitions(),r=rDelta())
    while S.need_r(confidence_level()):
        print "Extending r to", S.numbers_1xr.shape[1] + rDelta()
        S.extend_r(rDelta())
        if S.numbers_1xr.shape[1] > 210000:
            return np.inf
    return S.r_star(confidence_level())

def compute_one(p, n, n_alt_plus, n_alt_minus,):
    B = simple.Simple(n=n, p=p).flatten()
    Bplus = simple.Simple(n=n_alt_plus, p=p).flatten()
    Bminus = simple.Simple(n=n_alt_minus, p=p).flatten()
    print "Computing Plus Alternative"
    SPlus = compute_alt(B, Bplus)
    print "Computing Minus Alternative"
    SMinus = compute_alt(B, Bminus)
    return SPlus, SMinus

def compute(ps,ns,ns_alt_plus,ns_alt_minus,reps,fname):
    dim = (reps, len(ps), len(ns))
    P = np.zeros(dim)
    N = np.zeros(dim)
    NPlus = np.zeros(dim)
    NMinus = np.zeros(dim)
    rStarPlus = np.zeros(dim)
    rStarMinus = np.zeros(dim)
    k = 0
    initial = time.time()
    for pi, p in enumerate(ps):
        for ni, n in enumerate(ns):
            for r in range(reps):
                start = time.time()
                k = k + 1
                print "Computing (r, p, n) =", r, p, n, "(", r, pi, ni, "), of", P.shape
                rstar_plus1, rstar_minus1 = compute_one(p,n,ns_alt_plus[ni],ns_alt_minus[ni])
                P[r, pi, ni] = p
                N[r, pi, ni] = n
                NPlus[r, pi, ni] = ns_alt_plus[ni]
                NMinus[r, pi, ni] = ns_alt_minus[ni]
                rStarPlus[r, pi, ni] = rstar_plus1
                rStarMinus[r, pi, ni] = rstar_minus1
                output = open(fname, 'wb')
                pickle.dump( (P,N,NPlus, NMinus, rStarPlus, rStarMinus), output)
                output.close()
                print "Work Saved to ", fname
                end = time.time()
                print 'Iteration', k, 'of', P.shape[0]*P.shape[1]*P.shape[2], 'took'
                print end-start, 'seconds OR', (end-start)/60, 'minutes OR', (end-start)/(60*60), 'hours'
                print 'Total elapased Time:'
                print end-initial, 'seconds OR', (end-initial)/60, 'minutes OR', (end-initial)/(60*60), 'hours'
    return P, N, NPlus, NMinus, rStarPlus, rStarMinus

def run():
    return compute(pspace(),nspace(),nspace()+1,nspace()-1,nreps(),OutFileName())

def runx():
    return compute(pspace(),nxspace(),nxspace()+nxspace()/10,nxspace()-nxspace()/10,nreps(),XOutFileName())

def loadI():
    infile = open(InFileName(),'rb')
    return pickle.load(infile)

def loadX():
    infile = open(XInFileName(),'rb')
    return pickle.load(infile)

def parse_means(A):
    return A[0].mean(axis=0), A[1].mean(axis=0), A[2].mean(axis=0), \
           A[3].mean(axis=0), A[4].mean(axis=0), A[5].mean(axis=0)

def parse_stds(A):
    return  A[4].std(axis=0), A[5].std(axis=0)

def se_regions(pi):
    A = loadI()
    P, N, Nplus, Nminus, rStarPlus, rStarMinus = parse_means(A)
    SDplus, SDminus = parse_stds(A)
    plt.fill_between(N[pi,:], rStarPlus[pi,:] - SDplus[pi,:]/np.sqrt(A[0].shape[0]),
                              rStarPlus[pi,:] + SDplus[pi,:]/np.sqrt(A[0].shape[0]),
                     color='red', alpha=.3)
    plt.fill_between(N[pi,:], rStarMinus[pi,:] - SDminus[pi,:]/np.sqrt(A[0].shape[0]),
                              rStarMinus[pi,:] + SDminus[pi,:]/np.sqrt(A[0].shape[0]),
                     color='blue', alpha=.3)
    plt.show()

def fixE14():
    infile = open(OldInFileName(),'rb')
    P, N, Nplus, Nminus, rStarPlus, rStarMinus = pickle.load(infile)
    X = rStarPlus
    X[np.where(X>1e14)] = 1.
    return P, N, Nplus, Nminus, X, rStarMinus