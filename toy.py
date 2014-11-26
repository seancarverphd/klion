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
        self.clearData()  # Reseeds random number generator
        self.changeModel(parent)
    def clearData(self):
        self.nReps = None
        self.setSeed()
        self.data = []
        self.likes = []
        self.changedSinceLastSim = False
    def setSeed(self):
        if self.seed == None:
            self.usedSeed = long(time.time()*256)
        else:
            self.usedSeed = self.seed  # can be changed with self.reseed()
        self.R.seed(self.usedSeed)  # For simulating Markov Chain
        self.changedSinceLastSim = True
    def reseed(self,seed):
        self.seed = seed
        self.clearData()  # Reseeds random number generator
    def changeModel(self,parent):
        self.parseParams(parent)
        self.changedSinceLastSim = True ### ADD TO ENGINE!!!
        # Don't clearData(); might want to change Model and use old data 
    def parseParams(self,parent):  # For subclassing replace this code
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
    def sim(self,nReps=1,clear=False): # Only does new reps; keeps old; if (nReps < # Trajs) then does nothing
        if clear:
            self.clearData()  # Reseeds random number generator
        elif self.changedSinceLastSim:
            self.clearData()
        numNewReps = nReps - len(self.data)
        for n in range(numNewReps):  
            self.data.append(self.simulateOnce(self.R)) # Don't want to use self.R elsewhere
        self.nReps = nReps
        self.changedSinceLastSim = False
    def simulateOnce(self,RNG=None):  #Subclassing, replace: Don't pass self.R if you want to avoid changing its state
        R = self.getRandom(RNG)
        if self.toy2:
            return (R.expovariate(self.q))
        else:
            return (R.expovariate(self.q1)+R.expovariate(self.q0))
    def getRandom(self,RNG=None):  #RNG == None creates new Random Number Generator and sets seed to time
        if RNG == None:
            R = random.Random()
            R.seed(long(time.time()*256))
        else:
            R = RNG
        return R
    def likelihoods(self,data=None):
        if data == None:
            data = self.data[len(self.likes):self.nReps]
            likes = self.likes
            nLast = self.nReps
        else:
            likes = []
            nLast = len(data)
        for datum in data:
            likes.append(self.likeOnce(datum))
        return likes[0:nLast]
    def likeOnce(self,datum):  # Subclassing, replace
        if datum < 0.:
            return -numpy.infty
        elif self.toy2:
            return (numpy.log(self.q) - self.q*datum)
        elif self.q0 == self.q1:
            return (numpy.log(self.q1) + numpy.log(self.q0) - self.q0*datum + numpy.log(datum))
        elif datum == 0.:  # already know its toy3
            return -numpy.infty
        else:
            return (numpy.log(self.q1)+numpy.log(self.q0)+numpy.log((numpy.exp(-self.q0*datum)-numpy.exp(-self.q1*datum))/(self.q1-self.q0)))
        #if self.toy2:
        #    return(numpy.log(self.q) - self.q*datum)
        #else:
        #    return(numpy.log(self.q1) + numpy.log(self.q0) + numpy.log((numpy.exp(-self.q0*datum)-numpy.exp(-self.q1*datum))/(self.q1-self.q0)))
    def minuslike(self,data=None):
        L = self.likelihoods(data)
        return -sum(L)       
    def like(self,data=None):
        L = self.likelihoods(data)
        return sum(L)
    def pdf(self,datum):
        return numpy.exp(self.likelihoods([datum])[0])
        #if datum < 0.:
        #    return 0.
        #elif self.toy2:
        #    return numpy.exp(numpy.log(self.q) - self.q*datum)
        #elif self.q0 == self.q1:
        #    return numpy.exp(numpy.log(self.q1) + numpy.log(self.q0) - self.q0*datum + numpy.log(datum))
        #elif datum == 0.:  # toy3
        #    return 0.
        #else:
        #    return numpy.exp(numpy.log(self.q1)+numpy.log(self.q0)+numpy.log((numpy.exp(-self.q0*datum)-numpy.exp(-self.q1*datum))/(self.q1-self.q0)))
    def mle(self):
        assert(self.toy2)  # Not yet implemented for toy 3
        return 1./numpy.mean(self.data[0:self.nReps])
    def logf(self,data=None):
        return numpy.matrix(self.likelihoods(data))
        #if data == None:
        #    data = self.data[0:self.nReps]
        #data = numpy.matrix(data)
        #if self.toy2:
        #    return(numpy.log(self.q) - self.q*data)
        #else:
        #    return(numpy.log(self.q1) + numpy.log(self.q0) + numpy.log((numpy.exp(-self.q0*data)-numpy.exp(-self.q1*data))/(self.q1-self.q0)))
    def lr(self,alt):  # likelihood ratio; self is true model
        data = self.data[0:self.nReps]
        return (self.logf(data) - alt.logf(data))
    def lr_mn_sd(self,alt):  # self is true model
        lrs = self.lr(alt)
        mn = numpy.mean(lrs)
        sd = numpy.std(lrs)
        return (mn,sd)
    def lrN(self,alt,N,M):  # add N of them, return M
        self.sim(nReps=N*M)
        lrNM = self.lr(alt)
        L = numpy.reshape(lrNM,(M,N))
        return L.sum(axis=0)
    def aic(self,alt):  # self is true model
        data = self.data[0:self.nReps]
        return 2*(self.logf(data) - alt.logf(data))
    def a_mn_sd(self,alt):  # self is true model
        aics = self.aic(alt)
        mn = numpy.mean(aics)
        sd = numpy.std(aics)
        return (mn,sd)
    def aicN(self,alt,N,M):  # add N of them, return M
        self.sim(nReps=N*M)
        aicNM = self.aic(alt)
        A = numpy.reshape(aicNM,(M,N))
        return A.sum(axis=0)
    def Eflogf(self, data=None):  # NEED TO ADJUST FOR REPEATED EXPERIMENTS (M and N both different from 1)
        return(self.logf(data).mean())
        #if self.toy2:
        #    return numpy.log(self.q) - self.q*numpy.mean(self.data[0:self.nReps])
        #else:  # toy 3
        #    Qs = []
        #    for n in range(self.nReps):
        #        if self.q1 == self.q0:
        #            Qs.append(-self.q0*self.data[n] + numpy.log(self.data[n]))    
        #        else:
        #            Qs.append(numpy.log((numpy.exp(-self.q0*self.data[n])-numpy.exp(-self.q1*self.data[n]))/(self.q1-self.q0)))
        #            if numpy.isinf(Qs[-1]):
        #                print "q1", self.q1, "q0", self.q0
        #                
        #    Qbar = numpy.mean(Qs)
        #    return numpy.log(self.q1) + numpy.log(self.q0) + Qbar
    def Eflogg(self,data):  # Must pass data, don't use self.data
        return self.Eflogf(data)
        # assert(self.toy2)
        # return numpy.log(self.q) - self.q*numpy.mean(data)  # data passed as parameter: not self.data!
    #def pdfplot(self):
    #    assert(len(self.data)>99)
    #    m = min(self.data)
    #    M = max(self.data)
    #    X = numpy.arange(m,M,(M-m)/100)
    #    Y = []
    #    for x in X:
    #        Y.append(self.pdf(x))
    #    plt.plot(X,Y)
    #    plt.hist(self.data,50,normed=1)
    #    plt.show()

def toy3mlike4opt(q,data):
    for datum in data:
        self.mll -= ad.admath.log(q[1]) + ad.admath.log(q[0])
        self.mll -= ad.admath.log((ad.admath.exp(-q[0]*datum)-ad.admath.exp(-q[1]*datum))/(q[1]-q[0]))
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
