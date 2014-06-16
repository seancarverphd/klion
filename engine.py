import numpy
import math
import random

class flatStepProtocol(object):
    def __init__(self,seed):
        self.seed = seed
        self.R = random.Random()
        self.R.seed(seed)
    def appendTrajectory(self,state,time,volts):
        self.simStates.append(state)
        self.simDataT.append(time)
        # Might want to modify next line: multiply conductance by "voltage" to get current
        # I think "voltage" should really be difference between voltage and reversal potential
        self.simDataX.append(self.R.normalvariate(self.Mean[state],self.Std[state]))
        self.simDataV.append(volts)
    def initTrajectory(self,initDistrib,voltages,durations,Mean,Std,A,dt):
        self.initDistrib = initDistrib
        self.voltages = voltages
        self.durations = durations
        self.nsamples = []
        for dur in self.durations:
            self.nsamples.append(int(math.ceil(dur/dt)))
        # These might depend on voltage as well as state
        self.Mean = Mean
        self.Std = Std
        self.A = A
        self.simStates = []
        self.simDataT = []
        self.simDataX = []
        self.simDataV = []
        self.dt = dt
        state0 = self.select(self.initDistrib)
        self.appendTrajectory(state0,0.,voltages[0])
    def sim(self):
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
