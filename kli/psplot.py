__author__ = 'sean'

import numpy as np
import matplotlib.pylab as plt
import matplotlib
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

def se_regions(pi, rel=False, figax=None):
    if figax is None:
        figax = plt.subplots()
    if rel:
        A = loadX()
        plus_color = 'green'
        plus_alpha = .3
        minus_color = 'yellow'
        minus_alpha = .8
    else:
        A = loadI()
        plus_color = 'grey'
        plus_alpha = .3
        minus_color = 'red'
        minus_alpha = .3
    P, N, Nplus, Nminus, rStarPlus, rStarMinus = parse_means(A)
    SDplus, SDminus = parse_stds(A)
    figax[1].fill_between(N[pi,:], rStarPlus[pi,:] - SDplus[pi,:]/np.sqrt(A[0].shape[0]),
                              rStarPlus[pi,:] + SDplus[pi,:]/np.sqrt(A[0].shape[0]),
                     color=plus_color, alpha=plus_alpha)
    figax[1].fill_between(N[pi,:], rStarMinus[pi,:] - SDminus[pi,:]/np.sqrt(A[0].shape[0]),
                              rStarMinus[pi,:] + SDminus[pi,:]/np.sqrt(A[0].shape[0]),
                     color=minus_color, alpha=minus_alpha)
    figax[0].show()

def fixE14():
    infile = open(OldInFileName(),'rb')
    P, N, Nplus, Nminus, rStarPlus, rStarMinus = pickle.load(infile)
    X = rStarPlus
    X[np.where(X>1e14)] = 1.
    return P, N, Nplus, Nminus, X, rStarMinus

def venn(center, radius, distance, figax=None):
    if figax is None:
        figax = plt.subplots()
    plt.figure(figax[0].number)
    v = figax[1].axis()
    width = radius*(v[1]-v[0])
    height = radius*(v[3]-v[2])
    xdistance = np.sqrt(3.)/4.*distance*(v[1]-v[0])
    ydistance = distance*(v[3]-v[2])
    circle0 = matplotlib.patches.Ellipse(center,width,height,facecolor='yellow',alpha=.6,edgecolor='black')
    circle1 = matplotlib.patches.Ellipse((center[0]-xdistance,center[1]+ydistance),width,height,
                                         facecolor='red',alpha=.4,edgecolor='black')
    circle2 = matplotlib.patches.Ellipse((center[0]+xdistance,center[1]+ydistance),width,height,
                                         facecolor='blue',alpha=.4,edgecolor='black')
    figax[1].add_artist(circle0)
    figax[1].add_artist(circle1)
    figax[1].add_artist(circle2)
    figax[0].show()
    return figax

def venn2(center, radius, distance, figax=None):
    if figax is None:
        figax = plt.subplots()
    plt.figure(figax[0].number)
    v = figax[1].axis()
    width = radius*(v[1]-v[0])
    height = radius*(v[3]-v[2])
    xdistance = np.sqrt(3.)/2.*distance*(v[1]-v[0])
    ydistance = 2*distance*(v[3]-v[2])
    circle0 = matplotlib.patches.Ellipse(center,width,height,facecolor='red',alpha=.3,edgecolor='black')
    circle1 = matplotlib.patches.Ellipse((center[0]+xdistance,center[1]),width,height,
                                         facecolor='grey',alpha=.3,edgecolor='black')
    circle2 = matplotlib.patches.Ellipse((center[0],center[1]-ydistance),width,height,
                                         facecolor='yellow',alpha=.8,edgecolor='black')
    circle3 = matplotlib.patches.Ellipse((center[0]+xdistance,center[1]-ydistance),width,height,
                                        facecolor='green',alpha=.3,edgecolor='black')
    figax[1].add_artist(circle0)
    figax[1].add_artist(circle1)
    figax[1].add_artist(circle2)
    figax[1].add_artist(circle3)
    figax[0].show()

def my_subplots_for_sfn():
    fig = plt.figure()
    ax11 = plt.subplot(321)
    ax12 = plt.subplot(322)
    ax21 = plt.subplot(323,sharex=ax11,sharey=ax11)
    ax22 = plt.subplot(324,sharex=ax12)
    ax31 = plt.subplot(325,sharex=ax11,sharey=ax11)
    ax32 = plt.subplot(326,sharex=ax12)
    ax11.set_xlim(0,100)
    ax12.set_xlim(0,120)
    ax11.set_ylim(0,.14)
    ax12.set_ylim(0,140)
    ax22.set_ylim(0,1400)
    ax32.set_ylim(0,14000)
    return (fig,[[ax11,ax12], [ax21, ax22], [ax31,ax32]])

def SfNplot():
    ntrue = 100
    nalt0 = 99
    nalt1 = 90
    S100_1 = simple.Simple(n=ntrue,p=.1).flatten()
    S99_1 = simple.Simple(n=nalt0,p=.1).flatten()
    S90_1 = simple.Simple(n=nalt1,p=.1).flatten()
    S100_5 = simple.Simple(n=ntrue,p=.5).flatten()
    S99_5 = simple.Simple(n=nalt0,p=.5).flatten()
    S90_5 = simple.Simple(n=nalt1,p=.5).flatten()
    S100_9 = simple.Simple(n=ntrue,p=.9).flatten()
    S99_9 = simple.Simple(n=nalt0,p=.9).flatten()
    S90_9 = simple.Simple(n=nalt1,p=.9).flatten()
    fig, ax = my_subplots_for_sfn()
    S100_1.compare_3bars(S99_1,S90_1,ntrue,(fig,ax[2][0]),xlab='Number of open channels (k)')
    S100_5.compare_3bars(S99_5,S90_5,ntrue,(fig,ax[1][0]))
    S100_9.compare_3bars(S99_9,S90_9,ntrue,(fig,ax[0][0]))
    se_regions(2,False,(fig,ax[0][1]))
    se_regions(1,False,(fig,ax[1][1]))
    se_regions(0,False,(fig,ax[2][1]))
    se_regions(2,True,(fig,ax[0][1]))
    se_regions(1,True,(fig,ax[1][1]))
    se_regions(0,True,(fig,ax[2][1]))
    venn((30,.1),.1,.05,(fig,ax[0][0]))
    ax[0][0].text(18,.085,'n-10%=90')
    ax[0][0].text(41,.105,'n=100')
    ax[0][0].text(4.5,.105,'n-1=99')
    ax[0][0].text(5,.12,'Number of Channels:')
    ax[0][0].text(5,.13,'Probability of Opening: p=0.9')
    ax[1][0].text(5,.13,'Probability of Opening: p=0.5')
    ax[2][0].text(20,.13,'Probability of Opening: p=0.1')
    ax[0][1].text(5,130,'Probability of Opening: p=0.9')
    ax[0][1].text(5,120,'Number of Channels in Alternative:')
    ax[0][1].text(10,110,'(Falsified with 95% Confidence)')
    ax[1][1].text(5,1300,'Probability of Opening: p=0.5')
    ax[2][1].text(5,13000,'Probability of Opening: p=0.1')
    ax[0][1].set_ylabel('Needed Sample Size')
    ax[1][1].set_ylabel('Needed Sample Size')
    ax[2][1].set_ylabel('Needed Sample Size')
    ax[2][1].set_xlabel('Number of Channels in True Model (n)')
    ax[0][1].text(5,90,'n-1')
    ax[0][1].text(5,72.5,'n-10%')
    ax[0][1].text(43,90,'n+1')
    ax[0][1].text(43,72.5,'n+10%')
    venn2((29,90),.1,.06,(fig,ax[0][1]))
    fig.show()
    return fig, ax

def regress_plot():
    F = open('regressdata.pkl','rb')
    x,y,r,m,b = pickle.load(F)
    fig, ax = plt.subplots()
    plt.scatter(x,y)
    plt.plot(x,r)
    plt.xlim(x[0],x[-1])
    plt.ylim(.94,.96)
    ax_left, ax_right, ax_lo, ax_hi = plt.axis()
    region_below = matplotlib.patches.Rectangle((x[0],ax_lo), x[-1]-x[0],.95-ax_lo, color='red', alpha=.3)
    ax.add_patch(region_below)
    region_above = matplotlib.patches.Rectangle((x[0],0.95), x[-1]-x[0],ax_hi-.95, color='green', alpha=.3)
    ax.add_patch(region_above)
    plt.xlabel('Sample Size')
    plt.ylabel('Confidence Level')
    rstar = (0.95-b)/m
    plt.arrow(rstar,.948,0.,.0013,fc='k',ec='k',head_width=20,head_length=.0003)
    plt.text(rstar,.9474,'Estimate of Needed Sample Size')
    plt.text(9600,.9593,'True Model Has n=100 Channels')
    plt.text(9600,.9587,'Alternative Has 99 Channels')
    plt.text(9600,.9581,'Probability of Opening: p=0.1')
    plt.show()
    return fig, ax

