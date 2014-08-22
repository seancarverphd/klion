import numpy
import matplotlib
import matplotlib.pylab as plt
import parameter
import toy
import cPickle as pickle

def Ksave(K,fname):
    f = open(fname,'wb')
    pickle.dump(K.qRange,f)
    pickle.dump(K.KL,f)
    f.close()
    
def Kload(fname):
    f = open(fname,'rb')
    qRange = pickle.load(f)
    KL = pickle.load(f)
    f.close()
    K = Kshell(qRange,KL)
    return K

class Kshell(object):  # better name: Kskeleton
    def __init__(self,qRange,KL):
        self.qRange = qRange
        self.KL = KL
    def plot(self):
        cs = plt.pcolor(self.qRange,self.qRange,self.KL)
        cb = plt.colorbar(cs)
        plt.title('Three-State Model with Irreversible Transitions')
        plt.xlabel('First Rate Constant (kHz)')
        plt.ylabel('Second Rate Constant (kHz)')
        # Should save nReps and derive '10^5' from nReps=100000
        cb.set_label('Kullback-Leibler Divergence to 2-State Alternative\nEach Pixel: Monte-Carlo Integral with 10^5 Samples')
        plt.show()

class kull(object):
    def __init__(self,trueParent,altParent,q0=None,q1=None,q=None):
        self.q0 = q0
        self.q1 = q1
        self.q = q
        self.trueParent = trueParent
        self.altParent = altParent
        self.TrueMod = trueParent.flatten()
        self.AltMod = altParent.flatten()
        self.qRange = numpy.arange(.1,10,.1) # use for plotting
        #self.qRange = numpy.arange(1,10,2)  # use for testing
        self.nReps = 100000
    def compute(self):
        num = len(self.qRange)
        self.KL = numpy.zeros((num,num))
        for i,q_1 in enumerate(self.qRange):
            for j,q_0 in enumerate(self.qRange):
                self.q0.assign(q_0)
                self.q1.assign(q_1)
                self.TrueMod.changeModel(self.trueParent)
                self.TrueMod.sim(self.nReps,clear=True)
                EfTrue = self.TrueMod.Eflogf()
                mEfAlt = numpy.log(numpy.e*numpy.mean(self.TrueMod.taus))  # Simple because toy is simple
                self.KL[i,j] = EfTrue + mEfAlt
                print i, j, "up to", len(self.qRange)
    def save(self,fname):
        Ksave(self,fname)

class Nalpha(object):
    def __init__(self,trueParent,altParent,q0=None,q1=None,q=None):
        self.q0 = q0
        self.q1 = q1
        self.q = q
        self.trueParent = trueParent
        self.altParent = altParent
        self.trueModel = trueParent.flatten()
        self.altModel = altParent.flatten()
        #self.qRange = numpy.arange(.1,10,.1) # use for plotting
        self.qRange = numpy.arange(1,10,2) # use for testing
        self.MCSampSize = 1000
        self.numWrongTarget = 50
        self.numRightTarget = self.MCSampSize - self.NumWrongTarget # 1000-50==950
        self.alpha = self.numWrongTarget/self.MCSampSize # 50/1000 == 0.05
    def compute(self):
        num = len(self.qRange)
        self.NA = numpy.zeros((num,num))
        for i,q_1 in enumerate(self.qRange):
            for j,q_0 in enumerate(self.qRange):
                self.q0.assign(q_0)
                self.q1.assign(q_1)
                # Only do trueModel once per nRep step
                self.nReps = max_nReps
                # nReps set adaptively to achieve alpha 
                self.trueModel.changeModel(self.trueParent)
                self.trueModel.sim4aic(self.nReps,self.MCSampSize)
                # binary search
                while (not numRight==self.numRightTarget) and (nRep <= max_nReps and nReps >= 1):
                    # self.altModel.sim4aic(self.nReps,self.MCSampSize)
                    if numRight < self.numRightTarget:
                        nReps = int(floor(nReps*1.33))
                    elif numRight > self.numRightTarget:
                        nReps = int(floor(nReps/2))
                    # More to come ...
                # More to come ...
    def save(self,fname):
        NAsave(self,fname)
        
q0 = parameter.Parameter("q0",0.5,"kHz",log=True)
q1 = parameter.Parameter("q1",0.25,"kHz",log=True)
q = parameter.Parameter("q",1./6.,"kHz",log=True)
T3 = toy.toyProtocol([q0,q1])
T2 = toy.toyProtocol([q])
K = kull(T3,T2,q0,q1,q)
# Uncomment next three lines to generate 'largeKLDataSet.p' (untested); ToDo: Change code to save (& print in plot) nReps, and test
# K.compute()
# K.save('largeKLDataSet.p')
# print K.KL
KP = Kload('largeKLDataSet.p')  # This file not in repository; uncomment three lines above to generate; (needs testing)
KP.plot()

