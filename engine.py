import numpy
import math
import random
import time
import copy
import parameter
import pandas

class flatStepProtocol(object):
    def __init__(self,parent,seed=None):
        # Nothing below changes until indicated
        self.R = random.Random()
        self.dt = copy.copy(parameter.v(parent.dt))  # copy here and below may be redundant
        self.voltages = []
        self.durations = []  
        self.nsamples = [] 
        for v in parent.voltages:
            self.voltages.append(copy.copy(parameter.v(v)))  # deepcopy doesn't work with units
        for dur in parent.voltageStepDurations:
            self.durations.append(copy.copy(parameter.v(dur)))
            if dur == None:
                self.nsamples.append(None)
            else:
                self.nsamples.append(int(self.durations[-1]/self.dt))
        # levels: for NO-NOISE only: makes sense only for single channels or small ensembles
        self.levels = copy.copy(parent.thePatch.uniqueLevels) # This is a set, deepcopy fails in assert below (change)
        self.levelList = list(self.levels)
        self.voltageTrajectory()  # Only needed for plotting
        # Nothing above changes, but below can change
        self.seed = seed # changes with reseed()
        self.change(parent.thePatch)  # call change(newPatch) possible
        self.clearData()
    def change(self,newPatch):
        assert(newPatch.hasNoise==False)  # Later will implement NOISE
        assert(self.levels==newPatch.uniqueLevels)  # Only makes sense with NO-NOISE
        self.initDistrib = newPatch.equilibrium(self.voltages[0])  # deepcopy not necessary here, or with A
        self.nextDistrib = []
        for i, ns in enumerate(self.nsamples):        
            if ns == None:   # Requires new initialization of state when simulating
                self.nextDistrib.append(newPatch.equilibrium(self.voltages[i]))
        self.A = []  
        for v in self.voltages:
            self.A.append(newPatch.getA(v,self.dt))  # when change, getA() called with same v's, value can change
        self.states2levels(newPatch)
        self.makeB() # NO-NOISE only. For NOISE: Set MEAN and STD here
    def states2levels(self,newPatch):
        self.levelMap = []
        self.levelNum = []
        self.states = []
        self.means = []
        for n in newPatch.ch.nodes:
            self.states.append(n)  # saved for output
            for u in range(len(self.levelList)):
                if n.level is self.levelList[u]:
                    self.levelMap.append(self.levelList[u])
                    self.levelNum.append(u)
                    continue
        self.nStates = len(self.levelMap)  # changes
        assert(self.nStates==len(newPatch.ch.nodes))  # make sure 1 level appended per node
    def voltageTrajectory(self):
        self.simDataV = []
        self.simDataV.append(self.voltages[0])
        for i,ns in enumerate(self.nsamples):
            if ns == None:
                self.simDataV.append(self.voltages[i])   # only append one voltage here
                continue
            for j in range(ns):
               self.simDataV.append(self.voltages[i]) # append one voltage for each sample
    def clearData(self):
        self.nReps = None
        if self.seed == None:
            self.usedSeed = long(time.time()*256)
        else:
            self.usedSeed = self.seed  # changes with reseed()
        self.R.seed(self.usedSeed)
        self.simStates = []
        self.simDataL = []
    def makeNewTraj(self):
        simS = []
        simL = []
        state0 = self.select(self.initDistrib)
        initNum = 0
        self.appendTrajectory(state0,simS,simL)
        return (simS,simL,initNum)
    def reseed(self,seed):
        self.seed = seed
        self.clearData()
    def appendTrajectory(self,state,simS,simL):
        simS.append(state)
        # NO NOISE:
        simL.append(self.levelNum[state])  # use self.levelMap for actual levels (not nums)
        # NOISE: 
        # Might want to modify next line: multiply conductance by "voltage" to get current
        # where I think "voltage" should really be difference between voltage and reversal potential
        # self.simDataX.append(self.R.normalvariate(self.Mean[state],self.Std[state]))
        # self.simDataX.append(self.Mean[state])
        # IF SAVING VOLTAGE:
        # self.simDataV.append(volts)
    def nextInit(self,initNum,simS,simL):
        state0 = self.select(self.nextDistrib[initNum])
        self.appendTrajectory(state0,simS,simL)
        return state0
    def sim(self,nReps=1):
        for n in range(nReps - len(self.simDataL)):
            (simS,simL,initNum) = self.makeNewTraj()  # sets initNum=0
            state = simS[0]
            for i,ns in enumerate(self.nsamples):
                if ns == None:
                    state = self.nextInit(initNum,simS,simL)
                    initNum += initNum
                    continue
                for j in range(ns):
                    state = self.select(self.A[i],state)
                    self.appendTrajectory(state,simS,simL)
            self.simStates.append(simS)
            self.simDataL.append(simL)
        self.nReps = nReps
    def resim(self,nReps=1):
        self.clearData()
        self.sim(nReps)
    def dataFrame(self,n):  # strips units off for plotting; pyplot can't handle units
        time = 0*self.dt  # multiplying by 0 preseves units
        # Commented out lines below are for computing quantities without units
        # mdt = parameter.m(self.dt)
        # simDataTm = numpy.arange(0,mdt*len(self.simStates),mdt)
        simNodes = []
        simDataT = []
        simDataC = []
        for s in self.simStates[n]:
            simNodes.append(self.states[s])   # simNodes are Node classes; simStates are  integers
            simDataT.append(copy.copy(time))
            simDataC.append(parameter.v(self.levelMap[s].mean) )
            # simDataCm.append(parameter.m(self.levelMap[s].mean))
            time += self.dt
        #simDataVm = []
        #for v in self.simDataV:
        #    simDataVm.append(parameter.m(v))
        # simDataC taken out of DataFrame
        return(pandas.DataFrame({'Time':simDataT,'Node':simNodes,'Voltage':self.simDataV}))
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
    def update(self,distrib,k,n):
        new = distrib*self.B[self.simDataL[n][k]]  # n is trajectory number, k=0 is sample num; doesn't depend on voltage
        return self.normalize(new)
    def predictupdate(self,distrib,k,iv,n):
        new = distrib*self.AB[self.simDataL[n][k]][iv]  # [[traj num][level num]][voltage num]
        return self.normalize(new)
    def minuslike(self,reps):  # returns minus the log-likelihood
        if reps == None:
            reps = range(self.nReps)
        self.mll = 0
        for n in reps:
            (alphak,ck) = self.update(self.initDistrib,0,n)
            self.mll += math.log(ck)
            k0 = 1   # Offset
            for iv in range(len(self.voltages)):
                for k in range(k0,k0+self.nsamples[iv]):
                    (alphak,ck) = self.predictupdate(alphak,k,iv,n)
                    self.mll += math.log(ck)
                k0 += self.nsamples[iv]
        return self.mll
    def like(self,reps=None):  # returns the log-likelihood
        return -self.minuslike(reps)
