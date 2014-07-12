import numpy
import matplotlib
import matplotlib.pylab as plt

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
        