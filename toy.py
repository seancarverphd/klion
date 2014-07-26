import numpy
import random
import time
import parameter
import matplotlib.pylab as plt
import scipy.optimize as opt

preferred = parameter.preferredUnits()
preferred.time = 'ms'
preferred.freq = 'kHz'

class toyProtocol(object):
    def __init__(self,q):
        self.q = q
        self.preferred = preferred
    def flatten(self,seed=None):
        parent = self  # for readability
        FT = flatToyProtocol(parent,seed)
        return FT
    
class flatToyProtocol(object):
    def __init__(self, parent, seed=None):
        self.R = random.Random()
        self.seed = seed
        self.clearData()
        self.changeModel(parent)
    def clearData(self):
        self.nReps = None
        if self.seed == None:
            self.usedSeed = long(time.time()*256)
        else:
            self.usedSeed = self.seed  # can be changed with self.reseed()
        self.R.seed(self.usedSeed)  # For simulating Markov Chain
        self.taus = []
    def reseed(self,seed):
        self.seed = seed
        self.clearData()  # Reseeds random number generator
    def changeModel(self,parent):
        if len(parent.q) == 1:
            self.toy2 = True
            self.q = parameter.mu(parent.q[0],parent.preferred.freq)
            self.q0 = None
            self.q1 = None
        elif len(parent.q) == 2:
            self.toy2 = False
            self.q = None
            self.q0 = parameter.mu(parent.q[0],parent.preferred.freq)
            self.q1 = parameter.mu(parent.q[1],parent.preferred.freq)
        else:
            assert(False) # Length of q should be 1 or 2
        self.changedSinceLastSim = True ### ADD TO ENGINE!!!
        # Don't clearData(); might want to change Model and use old data 
    def sim(self,nReps=1,clear=False): # Only does new reps; keeps old; if (nReps < # Trajs) then does nothing
        if clear:
            self.clearData()  # Reseeds random number generator
        elif self.changedSinceLastSim:
            self.clearData()
        numNewReps = nReps - len(self.taus)
        for n in range(numNewReps):  
            if self.toy2:
                self.taus.append(self.R.expovariate(self.q))
            else:
                self.taus.append(self.R.expovariate(self.q1)+self.R.expovariate(self.q0))
        self.nReps = nReps
        self.changedSinceLastSim = False
    def minuslike(self): # Still need to implement MC Sampling maybe: whichReps = range(self.nReps) for default; Not before AD
        self.mll = 0.
        if self.toy2:
            for n in range(self.nReps):
                self.mll -= (numpy.log(self.q) - self.q*self.taus[n])
        elif self.q0 == self.q1:
            for n in range(self.nReps):
                self.mll -= numpy.log(self.q1) + numpy.log(self.q0) - self.q0*self.taus[n] + numpy.log(self.taus[n])
        else:
            for n in range(self.nReps):
                self.mll -= numpy.log(self.q1) + numpy.log(self.q0)
                self.mll -= numpy.log((numpy.exp(-self.q0*self.taus[n])-numpy.exp(-self.q1*self.taus[n]))/(self.q1-self.q0))
        return self.mll
    def like(self):
        return -self.minuslike()
    def pdf(self,tau): # Still need to implement MC Sampling maybe: whichReps = range(self.nReps) for default; Not before AD
        if self.toy2:
            return numpy.exp(numpy.log(self.q) - self.q*tau)
        elif self.q0 == self.q1:
            return numpy.exp(numpy.log(self.q1) + numpy.log(self.q0) - self.q0*tau + numpy.log(tau))
        else:
            return numpy.exp(numpy.log(self.q1)+numpy.log(self.q0)+numpy.log((numpy.exp(-self.q0*tau)-numpy.exp(-self.q1*tau))/(self.q1-self.q0)))
    def mle(self):
        assert(self.toy2)  # Not yet implemented for toy 3
        return 1./numpy.mean(self.taus[0:self.nReps])
    def Eflogf(self):  # NEED TO ADJUST FOR REPEATED EXPERIMENTS (M and N both different from 1)
        if self.toy2:
            return numpy.log(self.q) - self.q*numpy.mean(self.taus[0:self.nReps])
        else:  # toy 3
            Qs = []
            for n in range(self.nReps):
                if self.q1 == self.q0:
                    Qs.append(-self.q0*self.taus[n] + numpy.log(self.taus[n]))    
                else:
                    Qs.append(numpy.log((numpy.exp(-self.q0*self.taus[n])-numpy.exp(-self.q1*self.taus[n]))/(self.q1-self.q0)))
                    if numpy.isinf(Qs[-1]):
                        print "q1", self.q1, "q0", self.q0
                        
            Qbar = numpy.mean(Qs)
            return numpy.log(self.q1) + numpy.log(self.q0) + Qbar
    def Eflogg(self,taus):
        assert(self.toy2)
        return numpy.log(self.q) - self.q*numpy.mean(taus)  # taus passed as parameter: not self.taus!
    def pdfplot(self):
        assert(len(self.taus)>99)
        m = min(self.taus)
        M = max(self.taus)
        X = numpy.arange(m,M,(M-m)/100)
        Y = []
        for x in X:
            Y.append(self.pdf(x))
        plt.plot(X,Y)
        plt.hist(self.taus,50,normed=1)
        plt.show()

def toy3mlike4opt(q,taus):
    for tau in taus:
        self.mll -= ad.admath.log(q[1]) + ad.admath.log(q[0])
        self.mll -= ad.admath.log((ad.admath.exp(-q[0]*tau)-ad.admath.exp(-q[1]*tau))/(q[1]-q[0]))
    return self.mll

class likefun(object):
    def __init__(self,parent,paramTuple):
        self.parent = parent
        self.paramTuple = paramTuple
        self.F = self.parent.flatten()
    def set(self,valueTuple):
        for i,P in enumerate(self.paramTuple):
            P.assign(valueTuple[i])
        self.F.changeModel(self.parent)
    def setLog(self,valueTuple):
        for i,P in enumerate(self.paramTuple):
            P.assignLog(valueTuple[i])  # AssignLog so that assigned values can vary from -infty to infty
        self.F.changeModel(self.parent)
    def sim(self,XTrue,nReps=100,seed=None,log=True):
        self.XTrue = XTrue
        if log==True:
            self.setLog(XTrue)
        else:
            self.set(XTrue)
        self.F.reseed(seed)
        self.F.sim(nReps,clear=True)  # clear=True should now be redundant, but kept here for readability
    def minuslike(self,x):
        self.setLog(x)
            #if x[0] < 0. or x[1]<0:
            #    print "x is negative"
            #print "x[0], q[0]", x[0], q0.value
            #print "x[1], q[1]", x[1], q1.value
        return self.F.minuslike()
    def like(self,x,log=True):
        if log==True:
            self.setLog(x)
        else:
            self.set(x)
        return self.F.like()
    
class likefun1(object):   # One dimensional likelihood grid
    def __init__(self,parent,XParam,seed=None):
        self.parent = parent
        self.XParam = XParam
        self.F = self.parent.flatten(seed=seed)
    def setRange(self,XRange):
        self.XRange = XRange
    def set(self,X):
        self.XParam.assign(X)
        self.F.changeModel(self.parent)
    def sim(self,XTrue=15,nReps=100,seed=None):
        self.XTrue = XTrue
        self.set(XTrue)
        self.F.reseed(seed)
        self.F.sim(nReps,clear=True)   # clear=True should now be redundant, but kept here for readability
    def compute(self):
        self.llikes = []
        for x in self.XRange:
            self.set(x)
            self.llikes.append(self.F.like())
    def plot(self):
        plt.plot(self.XRange,self.llikes)
        plt.show()
    def addVLines(self):
        pass
    def replot(self,XTrue=15,nReps=100,seed=None):
        self.sim(XTrue=XTrue,nReps=nReps,seed=seed)
        self.compute()
        self.plot()

class likefun2(object):   # Two dimensional likelihood grid
    def __init__(self,parent,XParam,YParam):
        self.parent = parent
        self.XParam = XParam
        self.YParam = YParam

q0 = parameter.Parameter("q0",0.5,"kHz",log=True)
q1 = parameter.Parameter("q1",0.25,"kHz",log=True)
q = parameter.Parameter("q",1./6.,"kHz",log=True)
T3 = toyProtocol([q0,q1])
T2 = toyProtocol([q])
FT3 = T3.flatten(seed=3)
FT2 = T2.flatten(seed=3)
XRange = numpy.arange(0.1,30.1,1)
YRange = numpy.arange(0.11,30.11,1)  # Different values so rate constants remain unequal

#One dimensional likelihood plot with toy2 model
#plt.figure()
#LF2 = likefun1(T2,q)
#LF2.setRange(XRange)
#LF2.replot(XTrue=15.,seed=10,nReps=100)

#One-dimensional likelihood plot with toy3 modle
#plt.figure()
#LF3 = likefun1(T3,q0)
#LF3.setRange(XRange)
#LF3.replot(XTrue=15,seed=11,nReps=1000)
LF = likefun(T3,[q0,q1])
LF.sim((1.,2.),nReps=1000,seed=0,log=True)

#Histogram and PDFs
#plt.figure()
#FT3.sim(nReps=1000,clear=True)
#FT3.pdfplot()
#plt.figure()
#FT2.sim(nReps=1000,clear=True)
#FT2.pdfplot()
#plt.figure()
#q0.assign(2.)
#q1.assign(2.)
#FTE = T3.flatten(seed=4)
#FTE.sim(nReps=1000,clear=True)
#FTE.pdfplot()
