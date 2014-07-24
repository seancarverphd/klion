import numpy
import matplotlib
import matplotlib.pylab as plt
import parameter
import toy

class kull(object):
    def __init__(self,trueParent,altParent,q0=None,q1=None,q=None):
        self.q0 = q0
        self.q1 = q1
        self.q = q
        self.trueParent = trueParent
        self.altParent = altParent
        self.TrueMod = trueParent.flatten()
        self.AltMod = altParent.flatten()
        self.qRange = numpy.arange(.1,10,1)
        self.nReps = 10000
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

q0 = parameter.Parameter("q0",0.5,"kHz",log=True)
q1 = parameter.Parameter("q1",0.25,"kHz",log=True)
q = parameter.Parameter("q",1./6.,"kHz",log=True)
T3 = toy.toyProtocol([q0,q1])
T2 = toy.toyProtocol([q])
K = kull(T3,T2,q0,q1,q)
K.compute()
print K.KL
  
