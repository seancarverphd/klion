import channel
import numpy as np
import math
import random
import parameter
import scipy
import scipy.linalg
from parameter import u

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
    def initTrajectory(self,seed,firstState=None):
        if isinstance(seed,random.Random):  # if seed a random num generator
            self.R = seed
        else:
            self.R.seed(seed)
        self.simStates = []
        self.simDataT = []
        self.simDataX = []
        self.simDataV = []
        if firstState == None: # if firstState not passed, draw from equilibrium
            state = self.thePatch.select(self.thePatch.equilibrium(self.voltages[0]))
        else:
            state = firstState
        self.simStates.append(state)
        self.simDataT.append(0.)
        # Might want to modify next line: multiply conductance by "voltage" to get current
        # I think "voltage" should really be difference between voltage and reversal potential
        self.simDataX.append(self.R.normalvariate(self.thePatch.Mean[state],self.thePatch.Std[state]))
        self.simDataV.append(parameter.m(self.voltages[0]))
    def appendTrajectory(self,nextState,time,volts):
        self.simStates.append(nextState)
        self.simDataT.append(time)
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

        
class Patch(object):
    def __init__(self, channels):
        self.channels = channels
        self.assertOneChannel()  # temporary; defines self.ch as temp handle to the channel
        self.voltages = [channel.V0,channel.V1,channel.V2,channel.V1]  # repeat V1
        self.voltageStepDurations = [0*u.ms,default_tstop,default_tstop,default_tstop]  # default_tstop is a global parameter
        self.firstState = None # if None, draws firstState from equilibrium distribution
        self.dt = default_dt  # default_dt is a global parameter
        # ^^^ Repeated variables affect differentiation via chain rule
        self.R = random.Random()
        self.Mean = self.ch.makeMean()
        self.Std = self.ch.makeStd()
    def assertOneChannel(self):
        assert(len(self.channels) == 1)
        assert(self.channels[0][0] == 1)
        self.ch = self.channels[0][1]
        assert(isinstance(self.ch,channel.Channel))
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
        for col in range(mat.shape[1]):  # iterate over columns of mat
            rowsum += mat[row, col]  # row constant passed into select
            if p < rowsum:
                return col
        assert(False)
    def initTrajectory(self):
        self.simStates = []
        self.simDataT = []
        self.simDataX = []
        self.simDataV = []
        if self.firstState == None: # if firstState not passed draw from equilibrium
            state = self.select(self.equilibrium(self.voltages[0]))
        else:
            state = self.firstState
        self.simStates.append(state)
        self.simDataT.append(0.);
        self.simDataX.append(self.R.normalvariate(self.Mean[state],self.Std[state]))
        self.simDataV.append(parameter.m(self.voltages[0]))
    def appendTrajectory(self,nextState,time,volts):
        self.simStates.append(nextState)
        self.simDataT.append(time)
        self.simDataX.append(self.R.normalvariate(self.Mean[nextState],self.Std[nextState]))
        self.simDataV.append(volts)
    def sim(self,seed=None):
        if not seed == None:  # if seed not passed, don't initialize R
            self.R.seed(seed)
        self.initTrajectory()
        mag_dt = parameter.m(self.dt)
        for i in range(len(self.voltages)):
            volts = parameter.m(self.voltages[i])
            nsamples = int(math.ceil(self.voltageStepDurations[i]/parameter.v(self.dt)))
            assert(nsamples >= 0)
            eQ = self.geteQ(self.voltages[i])
            # The next for-loop does the simulation on states
            for j in range(nsamples-1):
                # self.simStates[-1] is the row of eQ to work with, selected as from equilibrium()
                nextState = self.select(eQ,self.simStates[-1])
                time = self.simDataT[-1] + mag_dt
                self.appendTrajectory(nextState,time,volts)

P = Patch([(1, channel.khh)])
P.sim(seed=2)
voltages = [channel.V0,channel.V1,channel.V2,channel.V1]  # repeat V1
voltageStepDurations = [0*u.ms,default_tstop,default_tstop,default_tstop]  # default_tstop is a global parameter
S = StepProtocol(P,voltages,voltageStepDurations)
S.initTrajectory(2)
S.sim()
