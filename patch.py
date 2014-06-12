import channel
import numpy as np
import math
import random
import parameter
import scipy
import scipy.linalg
from parameter import u
import matplotlib
import matplotlib.pyplot as pyplot

default_dt = parameter.Parameter("dt",0.05,"ms",log=True)
default_tstop = parameter.Parameter("tstop",20,"ms",log=True)

def equilQ(Q):
    (V,D) = np.linalg.eig(Q.T)   # eigenspace
    imin = np.argmin(np.absolute(V))  # index of 0 eigenvalue
    eigvect0 = D[:,imin]  # corresponding eigenvector
    return eigvect0.T/sum(eigvect0) # normalize and return (fixes sign)

class StepProtocol(object):
    def __init__(self, patch, voltages, voltageStepDurations):
        self.thePatch = patch
        self.voltages = voltages
        self.voltageStepDurations = voltageStepDurations
        self.dt = default_dt
        self.R = random.Random()
    def setSampleInterval(self,dt):
        assert(parameter.m(dt)>0)
        self.dt = dt
    def initRNG(self,rng):
        if isinstance(rng,random.Random):  # if 1st arg passed is a random num generator, then use it
            self.R = rng                   # allows multiple trajs to have same generator
        else:                              # if 1 arg an integer, use as a seed
            self.R.seed(rng)
    def initTrajectory(self,rng,firstState=None):
        self.initRNG(rng)
        self.simStates = []
        self.simDataT = []
        self.simDataX = []
        self.simDataV = []
        if firstState == None: # if firstState not passed, draw from equilibrium
            theState = self.thePatch.select(self.thePatch.equilibrium(self.voltages[0]))
        else:
            theState = firstState
        time = 0.
        volts = parameter.m(self.voltages[0])
        self.appendTrajectory(theState,time,volts)
    def appendTrajectory(self,nextState,time,volts):
        self.simStates.append(nextState)
        self.simDataT.append(time)
        # Might want to modify next line: multiply conductance by "voltage" to get current
        # I think "voltage" should really be difference between voltage and reversal potential
        self.simDataX.append(self.R.normalvariate(self.thePatch.Mean[nextState],self.thePatch.Std[nextState]))
        self.simDataV.append(volts)
    def sim(self):
        mag_dt = parameter.m(self.dt)
        for i in range(len(self.voltages)):
            volts = parameter.m(self.voltages[i])
            nsamples = int(math.ceil(self.voltageStepDurations[i]/parameter.v(self.dt)))
            assert(nsamples >= 0)
            eQ = self.thePatch.geteQ(self.voltages[i])
            # The next for-loop does the simulation on states
            for j in range(nsamples-1):
                # self.simStates[-1] is the row of eQ to work with, selected as from equilibrium()
                nextState = self.thePatch.select(eQ,self.simStates[-1])
                time = self.simDataT[-1] + mag_dt
                self.appendTrajectory(nextState,time,volts)
                
class RepeatedSteps(StepProtocol):
    def initTrajectory(self,rng,firstState=None):
        self.initRNG(rng)
        self.firstState = firstState
        self.trajs = []
        self.nReps = 0
    def appendTrajectory(self,nReps):
        for i in range(nReps):
            T = StepProtocol(self.thePatch,self.voltages,self.voltageStepDurations)
            T.initTrajectory(self.R,self.firstState)
            T.sim()
            self.trajs.append(T)
        self.nReps+=nReps
        assert(self.nReps==len(self.trajs))
    def sim(self,nReps):
        assert(len(self.trajs)==0)
        self.appendTrajectory(nReps)
        

class singleChannelPatch(object):
    def __init__(self, ch):
        self.ch = ch
        self.voltages = [channel.V0,channel.V1,channel.V2,channel.V1]  # repeat V1
        self.voltageStepDurations = [0*u.ms,default_tstop,default_tstop,default_tstop]  # default_tstop is a global parameter
        self.firstState = None # if None, draws firstState from equilibrium distribution
        self.dt = default_dt  # default_dt is a global parameter
        self.R = random.Random()
        self.Mean = self.ch.makeMean()
        self.Std = self.ch.makeStd()
    def getQ(self,volts):
        channel.VOLTAGE.remap(volts)
        return self.ch.makeQ()
    def geteQ(self,volts):
        Q = self.getQ(volts)
        eQ = scipy.linalg.expm(self.dt*Q)
        # assert sum of rows is row of ones to tolerance
        tol = 1e-7
        assert(np.amin(np.sum(eQ,axis=1))>1.-tol)
        assert(np.amax(np.sum(eQ,axis=1))<1.+tol)
        return eQ
    def equilibrium(self,volts):
        return equilQ(self.getQ(volts))
    def select(self,mat,row=0):  # select from matrix[row,:]
        p = self.R.random()
        rowsum = 0
        # cols should add to 1
        for col in range(mat.shape[1]):  # iterate over columns of mat
            rowsum += mat[row, col]  # row constant passed into select
            if p < rowsum:
                return col
        assert(False)

P = singleChannelPatch(channel.khh)
voltages = [channel.V0,channel.V1,channel.V2,channel.V1]  # repeat V1; repeated variables affect differentiation via chain rule
voltageStepDurations = [0*u.ms,default_tstop,default_tstop,default_tstop]  # default_tstop is a global parameter
S = StepProtocol(P,voltages,voltageStepDurations)
S.initTrajectory(2)
S.sim()
RS = RepeatedSteps(P,voltages,voltageStepDurations)
RS.initTrajectory(4)
RS.sim(3)
# pyplot.plot(S.simDataT,S.simDataX)
# pyplot.show()
