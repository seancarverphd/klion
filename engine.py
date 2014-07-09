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
        self.R = random.Random() # for simulation of states
        self.RG = random.Random() # for simulation of conductance, currently separate from state
        self.seed = seed # can be changed with self.reseed()
        self.changeProtocol(parent)  # calls self.clearData()
        self.changeModel(parent.thePatch)
    def changeProtocol(self, parent):
        self.dt = copy.copy(parameter.v(parent.dt))  # copy here and below may be redundant
        self.voltages = []
        self.durations = []  
        self.nsamples = []
        self.preferred = parent.preferred  # preferred units
        for v in parent.voltages:
            self.voltages.append(copy.copy(parameter.v(v)))  # deepcopy doesn't work with units
        for dur in parent.voltageStepDurations:
            durationValue = copy.copy(parameter.v(dur))
            self.durations.append(durationValue)
            if numpy.isinf(durationValue):
                self.nsamples.append(None)
            else:
                self.nsamples.append(int(durationValue/self.dt))
        # levels: for NO-NOISE only: makes sense only for single channels or small ensembles
        self.levels = copy.copy(parent.thePatch.uniqueLevels) # This is a set, deepcopy fails in assert below (change)
        self.levelList = list(self.levels)
        self.hasVoltTraj = False # hasVoltTraj used in self.voltageTrajectory()  # Only needed for plotting
        self.clearData()
    def clearData(self):
        self.nReps = None
        if self.seed == None:
            self.usedSeed = long(time.time()*256)
        else:
            self.usedSeed = self.seed  # can be changed with self.reseed()
        self.R.seed(self.usedSeed)
        self.simStates = []
        self.simDataL = []
        self.simDataG = []  # conductance, simulated separately.
    def changeModel(self,newPatch):
        assert(newPatch.hasNoise==False)  # Later will implement NOISE
        assert(self.levels==newPatch.uniqueLevels)  # Only makes sense with NO-NOISE
        self.nextDistrib = []
        initDistrib = newPatch.ch.weightedDistrib()
        if initDistrib == None:   # No initial distribution because all weights 0, use equilibrium distribution
            assert(self.nsamples[0]==None)  #Infinite duration for first voltage, use equilibrium
        else:
            assert(not self.nsamples[0]==None)  # Finite duration for first voltage
            self.nextDistrib.append(initDistrib)  # Use initDistrib for initial distribution
        # OLD:  self.initDistrib = newPatch.equilibrium(self.voltages[0])  # deepcopy not necessary here, or with A
        for i, ns in enumerate(self.nsamples):        
            if ns == None:   # Requires new initialization of state when simulating
                self.nextDistrib.append(newPatch.equilibrium(self.voltages[i]))
        self.A = []  
        for v in self.voltages:
            self.A.append(newPatch.getA(v,self.dt))  # when change, getA() called with same v's, value can change
        self.states2levels(newPatch)
        self.makeB() # NO-NOISE only.
        self.makeMeanSTD()
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
        # The voltageTrajectory only depends on the Protocol not model.
        if self.hasVoltTraj:  # changeProtocol sets this to False
            return
        self.voltagesM = [] # Strip units off voltages
        for v in self.voltages:
            self.voltagesM.append(parameter.mu(v,self.preferred.voltage))
        self.simDataV = []
        self.simDataVM = []
        self.simDataT = []
        self.simDataTM = []
        dtValue = parameter.v(self.dt)
        dtValueM = parameter.mu(self.dt,self.preferred.time)  # without units
        #self.simDataV.append(self.voltages[0])
        for i,ns in enumerate(self.nsamples):  # one nsample per voltage, so iterates over voltages
            if ns == None:
                time = 0.*dtValue  # to define units
                timeM = 0 # no units
                self.simDataT.append(numpy.nan)  # time is an infinite interval here
                self.simDataTM.append(numpy.nan)
                self.simDataV.append(self.voltages[i])   # only append one voltage here
                self.simDataVM.append(self.voltagesM[i])
                continue
            elif i==0:  # not ns==None and i==0
                time = 0.*dtValue  # to define units
                timeM = 0 # no units
                self.simDataT.append(time)
                self.simDataTM.append(timeM)
                self.simDataV.append(numpy.nan)
                self.simDataVM.append(numpy.nan)
            for j in range(ns):
                time = copy.copy(time) + dtValue
                timeM += dtValueM
                self.simDataT.append(time)
                self.simDataTM.append(timeM)
                self.simDataV.append(self.voltages[i]) # append one voltage for each sample
                self.simDataVM.append(self.voltagesM[i])  # same voltage every sample until voltage steps
        hasVoltTraj = True
    def nextInit(self,nextInitNum):  # initializes state based on stored equilibrium distributions
        return self.select(self.nextDistrib[nextInitNum])
    def reseed(self,seed):
        self.seed = seed
        self.clearData()  # Reseeds random number generator
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
    def sim(self,nReps=1):
        for n in range(nReps - len(self.simDataL)):  # Only does new reps if nReps has increased since las 
            simS = []
            simL = []
            #state = self.makeNewTraj()  # sets initNum=0, initial init not counted
            #self.appendTrajectory(state,simS,simL)
            nextInitNum = 0
            for i,ns in enumerate(self.nsamples):
                if i==0 or ns == None:
                    state = self.nextInit(nextInitNum)
                    self.appendTrajectory(state,simS,simL)
                    nextInitNum += 1
                    continue
                for j in range(ns):
                    state = self.select(self.A[i],state)
                    self.appendTrajectory(state,simS,simL)
            self.simStates.append(simS)
            self.simDataL.append(simL)
        self.nReps = nReps
    def resim(self,nReps=1):
        self.clearData()  # reseeds random number generator
        self.sim(nReps)
    def makeMeanSTD(self):
        self.stds = []
        self.stdsM = []
        self.means = []
        self.meansM = []
        for s in self.states:
            self.means.append(parameter.v(s.level.mean))
            self.stds.append(parameter.v(s.level.std))
            self.meansM.append(parameter.mu(self.means[-1],self.preferred.conductance))
            self.stdsM.append(parameter.mu(self.stds[-1],self.preferred.conductance))
    #def simG(self,seedG=0):
    #    for SimS in simStates:   # simStates is a list of trajectories one for each rep.     
    #        for s in SS:        
    #            # self.simDataX.append(self.R.normalvariate(self.Mean[state],self.Std[state]))
    def dataFrame(self,rep=0, downsample=0,wantUnits=True):
        self.voltageTrajectory()
        means = []
        meansM = []
        for s in self.states:
            means.append(parameter.v(s.level.mean))
            meansM.append(parameter.mu(means[-1],self.preferred.conductance))
        DFNodes = []
        DFDataT = []
        DFDataG = []  # G is standard letter for conductance
        DFDataV = []
        counter = 0  # The counter is for downsampling
        if wantUnits:
            timeData = self.simDataT
            voltData = self.simDataV
            meanData = means
            TLabel = 'Time'
            VLabel = 'Voltage'
            GLabel = 'Conductance'
        else:
            timeData = self.simDataTM
            voltData = self.simDataVM
            meanData = meansM
            TLabel = 'T_'+self.preferred.time
            VLabel = 'V_'+self.preferred.voltage
            GLabel = 'G_'+self.preferred.conductance
        for i,s in enumerate(self.simStates[rep]):
            if numpy.isnan(self.simDataT[i]):  # reset counter with initialization (hold at pre-voltage)
                counter = downsample
            if counter >= downsample:   # Grab a data point
                counter = 0
                DFNodes.append(self.states[s])   # self.states are Node classes; s (in self.simStates) is an integer
                DFDataT.append(timeData[i])
                DFDataV.append(voltData[i])
                DFDataG.append(meanData[s])
            counter += 1
        dataDict = {TLabel:DFDataT,'Node':DFNodes,VLabel:DFDataV,GLabel:DFDataG}
        return(pandas.DataFrame(dataDict,columns=[TLabel,'Node',VLabel,GLabel]))           
    def select(self,mat,row=0):  # select from matrix[row,:]
    # select is also defined in patch.singleChannelPatch
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
