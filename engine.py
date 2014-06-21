import numpy
import math
import random
import parameter

class flatStepProtocol(object):
    def __init__(self,parent,seed=0):
        # Nothing below changes until indicated
        self.R = random.Random()
        self.dt = parameter.m(parent.dt) 
        self.voltages = []
        self.unitsVoltages = []
        self.durations = []  
        self.nsamples = [] 
        for v in parent.voltages:
            self.unitsVoltages.append(v)
            self.voltages.append(parameter.m(v))
        for dur in parent.voltageStepDurations:
            self.durations.append(parameter.m(dur))
            self.nsamples.append(int(math.ceil(parameter.m(dur)/self.dt)))
        # levels: for NO-NOISE only: makes sense only for single channels or small ensembles
        self.levels = parent.thePatch.uniqueLevels # This is a set
        self.levelList = list(self.levels)
        # Nothing above changes, but below can change
        self.seed = seed  # changes with reseed()
        self.change(parent.thePatch)  # call change(newPatch) possible
    def change(self,newPatch):
        assert(newPatch.hasNoise==False)  # Later will implement NOISE
        assert(self.levels==newPatch.uniqueLevels)  # Only makes sense with NO-NOISE
        self.initDistrib = newPatch.equilibrium(self.unitsVoltages[0])
        self.A = []  
        for v in self.unitsVoltages:
            self.A.append(newPatch.getA(v,self.dt))  # getA() called with same args, value can change
        self.states2levels(newPatch)
        self.makeB() # NO-NOISE only. For NOISE: Set MEAN and STD here
        self.clearData()
    def states2levels(self,newPatch):
        self.levelMap = []
        self.levelNum = []
        for n in newPatch.ch.nodes:
            for u in range(len(self.levelList)):
                if n.level is self.levelList[u]:
                    self.levelMap.append(self.levelList[u])
                    self.levelNum.append(u)
                    continue
        self.nStates = len(self.levelMap)  # changes
        assert(self.nStates==len(newPatch.ch.nodes))  # make sure 1 level appended per node
    def clearData(self):
        self.R.seed(self.seed)
        self.state0 = self.select(self.initDistrib)
        self.simStates = []
        self.simDataT = []
        # self.simDataX = []
        self.simDataL = []
        self.simDataV = []
        self.appendTrajectory(self.state0,0.,self.voltages[0])
        self.hasData=False
    def reseed(self,seed):
        self.seed = seed
        self.clearData()
    def appendTrajectory(self,state,time,volts):
        self.simStates.append(state)
        self.simDataT.append(time)
        # Might want to modify next line: multiply conductance by "voltage" to get current
        # where I think "voltage" should really be difference between voltage and reversal potential
        # NOISE: 
        # self.simDataX.append(self.R.normalvariate(self.Mean[state],self.Std[state]))
        # self.simDataX.append(self.Mean[state])
        # NO NOISE:
        self.simDataL.append(self.levelNum[state])  # use self.levelMap for actual levels (not nums)
        self.simDataV.append(volts)
    def sim(self):
        assert(self.hasData==False)
        time = 0
        for i in range(len(self.voltages)):
            for j in range(self.nsamples[i]):
                nextState = self.select(self.A[i],self.simStates[-1])
                time += self.dt
                self.appendTrajectory(nextState,time,self.voltages[i])
        self.hasData = True
    def resim(self):
        self.clearData()
        self.sim()
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
    def makeB(self):  # Only good for no-noise
        self.B = []
        self.AB = []
        for uniqueLevel in range(len(self.levels)):
            # Blevel is the B-matrix for the observation of level==uniqueLevel
            Blevel = numpy.zeros([self.nStates,self.nStates])
            for d in range(self.nStates):  # Fill B with corresponding 1's
                if self.levelNum[d]==uniqueLevel:
                    Blevel[d,d] = 1
            self.B.append(Blevel)
            # ABlevel is AB-matricies for all voltage steps, at given level
            ABlevel = []
            # AVolt is A-matrix for given voltage
            for Avolt in self.A:
                ABlevel.append(Avolt.dot(Blevel))
            self.AB.append(ABlevel)
    def normalize(self, new):
        c = 1/new.sum()
        return (c*new,c)
    def update(self,distrib,k):
        new = distrib*self.B[self.simDataL[k]]
        return self.normalize(new)
    def predictupdate(self,distrib,k,iv):
        new = distrib*self.AB[self.simDataL[k]][iv]  # [level num][voltage num]
        return self.normalize(new)
    def minuslike(self):  # returns minus the log-likelihood
        assert(self.hasData)
        (alphak,ck) = self.update(self.initDistrib,0)
        self.mll = math.log(ck)
        k0 = 1   # Offset
        for iv in range(len(self.voltages)):
            for k in range(k0,k0+self.nsamples[iv]):
                (alphak,ck) = self.predictupdate(alphak,k,iv)
                self.mll += math.log(ck)
            k0 += self.nsamples[iv]
        return self.mll
    def like(self):  # returns the log-likelihood
        return -self.minuslike()