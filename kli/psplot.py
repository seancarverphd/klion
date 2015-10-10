__author__ = 'sean'

import numpy as np

def pspace():
    return np.arange(.1, 1., .1)
    # brief: return np.arange(.1, 1., .4)

def nspace():
    return np.array([int(np.floor(x)) for x in np.exp(np.arange(.6,5,.3))])
    # brief: return np.arange(10, 100, 40)

def nxspace():
    return np.array([int(np.floor(x/10)*10) for x in np.exp(np.arange(2.9,7.2,.3))])
    # brief: return np.arange(10, 100, 40)

def nreps():
    return 9
    # brief: return 2

def compute_one(p, n, n_alt_plus, n_alt_minus):
    pass

def compute(ps,ns,ns_alt_plus,ns_alt_minus,reps):
    for p in ps:
        for i, n in enumerate(ns):
            for r in range(reps):
                star_plus, star_minus = compute_one(p,n,ns_alt_plus[i],ns_alt_minus[i])


def run():
    P,N,rStarPlus,rStarMinus = compute(pspace(),nspace(),nspace()+1,nspace()-1,nreps())

def runx():
    P,N,rStarPlus,rStarMinus = compute(pspace(),nxspace(),nxspace()+nxspace()/10,nxspace()-nxspace()/10,nreps())
