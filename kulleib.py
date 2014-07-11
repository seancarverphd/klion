import numpy
import matplotlib
import matplotlib.pylab as plt

class likegrid1(object):   # One dimensional likelihood grid
    def __init__(self,parent,XParam):
        self.parent = parent
        self.XParam = XParam
    def setRange(self,XRange):
        self.XRange = XRange
    def sim(self,XTrue=15,nReps=100,seed=None):
        self.XParam.assign(XTrue)
        self.F = self.parent.flatten(seed=seed)
        #print self.F.q0, self.F.q1
        self.F.sim(nReps,clear=True)
    def compute(self):
        self.llikes = []
        for x in self.XRange:
            self.XParam.assign(x)
            self.F.changeModel(self.parent)
            #print self.F.q0, self.F.q1
            self.llikes.append(self.F.like())
    def plot(self):
        plt.plot(self.XRange,self.llikes)
        plt.show()
    def replot(self,XTrue=15,nReps=100,seed=None):
        self.sim(XTrue=XTrue,nReps=nReps,seed=seed)
        self.compute()
        self.plot()
class likegrid2(object):   # Two dimensional likelihood grid
    def __init__(self,parent,XParam,YParam):
        self.parent = parent
        self.XParam = XParam
        self.YParam = YParam

class kull(object):
    def __init__(self,trueParent,altParent):
        self.P = trueParent.flatten()
        self.setNM(10,100)
    def setNM(self,N,M):
        self.nRepsPerE = N
        self.nMCSample = M
    def sim(self):
        if self.P.nReps < self.nRepsPerE*self.nMCSample:
            self.P.sim(nReps=self.nRepsPerE*self.nMCSample)
    def compute(self):
        self.sim()
        