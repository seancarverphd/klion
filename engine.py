import numpy
import math
import random
import parameter

class flatStepProtocol(object):
    def __init__(self,parent,seed):
        self.seed = seed
        self.R = random.Random()
        self.R.seed(seed)
        self.initDistrib = parent.thePatch.equilibrium(parent.voltages[0])
        self.voltages = []
        self.A = []
        for v in parent.voltages:
            self.voltages.append(parameter.m(v))
            self.A.append(parent.thePatch.getA(v,parent.dt))
        self.durations = []
        self.nsamples = []
        self.dt = parameter.m(parent.dt)
        for dur in parent.voltageStepDurations:
            self.durations.append(parameter.m(dur))
            self.nsamples.append(int(math.ceil(parameter.m(dur)/self.dt)))
        self.levels = parent.thePatch.ch.uniqueLevels
        self.levelMap = parent.thePatch.ch.levelMap
        self.nStates = len(self.levelMap)
        self.state0 = self.select(self.initDistrib)
        # NOISE: These might depend on voltage as well as state
        # theMean = []
        # theStd = []
        # for s in range(len(self.thePatch.ch.nodes)):
            # theMean.append(self.thePatch.Mean[s])
            # theStd.append(self.thePatch.Std[s])
    def reInit(self):
        self.simStates = []
        self.simDataT = []
        # self.simDataX = []
        self.simDataL = []
        self.simDataV = []
        self.appendTrajectory(self.state0,0.,self.voltages[0])
    def appendTrajectory(self,state,time,volts):
        self.simStates.append(state)
        self.simDataT.append(time)
        # Might want to modify next line: multiply conductance by "voltage" to get current
        # where I think "voltage" should really be difference between voltage and reversal potential
        # NOISE: 
        # self.simDataX.append(self.R.normalvariate(self.Mean[state],self.Std[state]))
        # self.simDataX.append(self.Mean[state])
        # NO NOISE:
        self.simDataL.append(self.levelMap[state])
        self.simDataV.append(volts)
    def sim(self):
        self.reInit()
        time = 0
        for i in range(len(self.voltages)):
            for j in range(self.nsamples[i]-1):  # Why the -1?
                nextState = self.select(self.A[i],self.simStates[-1])
                time += self.dt
                self.appendTrajectory(nextState,time,self.voltages[i])
    # Select is also defined in patch.singleChannelPatch
    def select(self,mat,row=0):  # select from matrix[row,:]
        p = self.R.random()
        rowsum = 0
        # cols should add to 1
        for col in range(mat.shape[1]):  # iterate over columns of mat
            rowsum += mat[row, col]  # row constant passed into select
            if p < rowsum:
                return col
        assert(False) # Should never reach this point
    def makeB(self):
        self.B = []
        self.AB = []
        for uniqueLevel in self.levels:
            Blevel = numpy.zeros([self.nStates,self.nStates])
            for d in range(self.nStates):
                if self.levelMap[d] is uniqueLevel:
                    Blevel[d,d] = 1
            self.B.append(Blevel)
            ABs = []
            for A in self.A:
                ABs.append(A.dot(Blevel))
            self.AB.append(ABs)
    def fit(self):
        for v in range(self.A):
            for k in self.nsamples:
                pass
            