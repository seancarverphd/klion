__author__ = 'sean'

import numpy as np
import star
import simple

def pspace():
    # heatmap: return np.arange(.1, 1., .1)
    # brief: return np.arange(.1, 1., .4)
    # briefer:
    return np.arange(.1, 1., .8)

def nspace():
    # return np.array([int(np.floor(x)) for x in np.exp(np.arange(.6,5,.3))])
    # brief:
    return np.arange(10, 100, 80)

def nxspace():
    # return np.array([int(np.floor(x/10)*10) for x in np.exp(np.arange(2.9,7.2,.3))])
    # brief:
    return np.arange(10, 100, 80)

def nreps():
    # return 9
    # brief:
    return 2

def mRepetitions():
    # return 200000
    # brief:
    return 1000

def kRepetitions():
    # return 10000
    # brief:
    return 1000

def rDelta():
    return 1000

def confidence_level():
    return 0.95

def compute_alt(B, Balt):
    S = star.Star(B,Balt,mReps=mRepetitions())
    S.root_table(k=kRepetitions(),r=rDelta())
    while S.need_r(confidence_level()):
        S.extend_r(rDelta())
    return S.r_star(confidence_level())

def compute_one(p, n, n_alt_plus, n_alt_minus,):
    B = simple.Simple(n=n, p=p).flatten()
    Bplus = simple.Simple(n=n_alt_plus, p=p).flatten()
    Bminus = simple.Simple(n=n_alt_minus, p=p).flatten()
    return compute_alt(B, Bplus), compute_alt(B, Bminus)

def compute(ps,ns,ns_alt_plus,ns_alt_minus,reps):
    dim = (reps, len(ps), len(ns))
    P = np.zeros(dim)
    N = np.zeros(dim)
    NPlus = np.zeros(dim)
    NMinus = np.zeros(dim)
    rStarPlus = np.zeros(dim)
    rStarMinus = np.zeros(dim)
    for pi, p in enumerate(ps):
        for ni, n in enumerate(ns):
            for r in range(reps):
                rstar_plus1, rstar_minus1 = compute_one(p,n,ns_alt_plus[ni],ns_alt_minus[ni])
                P[r, pi, ni] = p
                N[r, pi, ni] = n
                NPlus[r, pi, ni] = ns_alt_plus[ni]
                NMinus[r, pi, ni] = ns_alt_minus[ni]
                rStarPlus[r, pi, ni] = rstar_plus1
                rStarMinus[r, pi, ni] = rstar_minus1
    return P, N, NPlus, NMinus, rStarPlus, rStarMinus

def run():
    return compute(pspace(),nspace(),nspace()+1,nspace()-1,nreps())

def runx():
    return compute(pspace(),nxspace(),nxspace()+nxspace()/10,nxspace()-nxspace()/10,nreps())
