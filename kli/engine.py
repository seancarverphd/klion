import numpy
import math
import random
import time
import copy
import parameter
import pandas
import toy


class flatStepProtocol(toy.flatToyProtocol):
    def initRNG(self, seed):
        return toy.MultipleRNGs(2,seed) # random.Random()  # for simulation of states

    def restart(self):  # Clears data and resets RNG with same seed
        super(flatStepProtocol, self).restart()
        self.simDataGM = []  # conductance, simulated separately.

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
                self.nsamples.append(int(durationValue / self.dt))
        # levels: for NO-NOISE only: makes sense only for single channels or small ensembles
        self.levels = copy.copy(parent.thePatch.uniqueLevels)  # This is a set, deepcopy fails in assert below (change)
        self.levelList = list(self.levels)
        self.hasVoltTraj = False  # hasVoltTraj used in self.voltageTrajectory()  # Only needed for plotting
        self.restart()

    def changeModel(self, parent):
        newPatch = parent.thePatch
        assert (newPatch.hasNoise == False)  # Later will implement NOISE
        assert (self.levels == newPatch.uniqueLevels)  # Only makes sense with NO-NOISE
        self.nextDistrib = []
        initDistrib = newPatch.ch.weightedDistrib()
        if initDistrib == None:  # No initial distribution because all weights 0, use equilibrium distribution
            assert (self.nsamples[0] == None)  # Infinite duration for first voltage, use equilibrium
        else:
            assert (not self.nsamples[0] == None)  # Finite duration for first voltage
            self.nextDistrib.append(initDistrib)  # Use initDistrib for initial distribution
        # OLD:  self.initDistrib = newPatch.equilibrium(self.voltages[0])  # deepcopy not necessary here, or with A
        for i, ns in enumerate(self.nsamples):
            if ns == None:  # Requires new initialization of state when simulating
                self.nextDistrib.append(newPatch.equilibrium(self.voltages[i]))
        self.A = []
        for v in self.voltages:
            self.A.append(newPatch.getA(v, self.dt))  # when change, getA() called with same v's, value can change
        self.states2levels(newPatch)
        self.makeB()  # NO-NOISE only.
        self.makeMeanSTD()
        self.changedSinceLastSim = True
        # ??? Don't restart(); might want to change Model and use old data

    def states2levels(self, newPatch):
        self.levelMap = []
        self.levelNum = []
        self.nodes = []
        self.means = []
        for n in newPatch.ch.nodes:
            self.nodes.append(n)  # saved for output
            for u in range(len(self.levelList)):
                if n.level is self.levelList[u]:
                    self.levelMap.append(self.levelList[u])
                    self.levelNum.append(u)
                    continue
        self.nStates = len(self.levelMap)  # changes
        assert (self.nStates == len(newPatch.ch.nodes))  # make sure 1 level appended per node

    def voltageTrajectory(self):
        # The voltageTrajectory only depends on the Protocol not model.
        if self.hasVoltTraj:  # changeProtocol sets this to False
            return
        self.voltagesM = []  # Strip units off voltages
        for v in self.voltages:
            self.voltagesM.append(parameter.mu(v, self.preferred.voltage))
        self.simDataVM = []
        self.simDataTM = []
        dtValue = parameter.v(self.dt)
        dtValueM = parameter.mu(self.dt, self.preferred.time)  # without units
        # self.simDataV.append(self.voltages[0])
        for i, ns in enumerate(self.nsamples):  # one nsample per voltage, so iterates over voltages
            if ns == None:
                timeM = 0  # no units, M is for magnitude (no units)
                self.simDataTM.append(numpy.nan)
                self.simDataVM.append(self.voltagesM[i])
                continue
            elif i == 0:  # not ns==None and i==0
                timeM = 0  # no units
                self.simDataTM.append(timeM)
                self.simDataVM.append(numpy.nan)
            for j in range(ns):
                timeM += dtValueM
                self.simDataTM.append(timeM)
                self.simDataVM.append(self.voltagesM[i])  # same voltage every sample until voltage steps
        self.hasVoltTraj = True

    def nextInit(self, RNG, nextInitNum):  # initializes state based on stored equilibrium distributions
        return self.select(RNG, self.nextDistrib[nextInitNum])

    def appendTrajectory(self, state, simS, simL):
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

    def simulateOnce(self, RNG=None):
        if RNG is None:
            RNG = self.initRNG(None)
        simS = []
        simL = []
        nextInitNum = 0
        for i, ns in enumerate(self.nsamples):  # one nsample for each voltage step, equal number of samples in step
            if i == 0 or ns == None:  # if nsamples == None then indicates an initialization at equilibrium distrib
                state = self.nextInit(RNG.RNGs[0], nextInitNum)  # Next: append state and level to simS and simL
                self.appendTrajectory(state, simS, simL)  # Pass ref to simS & simL so that appendTrajectory works
                nextInitNum += 1
                continue
            for j in range(ns):  # Next i (could follow intializatation or another voltage step without init)
                state = self.select(RNG.RNGs[0], self.A[i], state)
                self.appendTrajectory(state, simS, simL)  # Pass ref to simS & simL so that appendTrajectory works
        self.recentState = simS
        # self.simStates.append(simS)
        # self.simDataL.append(simL)
        return simL

    def makeMeanSTD(self):
        self.stdsM = []
        self.meansM = []
        for s in self.nodes:
            newMean = parameter.v(s.level.mean)
            newSTD = parameter.v(s.level.std)
            self.meansM.append(parameter.mu(newMean, self.preferred.conductance))
            self.stdsM.append(parameter.mu(newSTD, self.preferred.conductance))

    def simG(self, nReps=1, clear=False):
        if clear:
            self.restart()
            # self.clearData()
        self.sim(nReps)  # Generates state trajectories, if needed
        for n in range(nReps - len(self.simDataGM)):
            newG = []
            for node in self.nodes[n]:
                newG.append(self.R.RNGs[1].normalvariate(self.meansM[node], self.stdsM[node]))
            self.simDataGM.append(newG)

    def dataFrame(self, rep=0, downsample=0):
        self.voltageTrajectory()
        hasG = (rep < len(self.simDataGM))
        DFNodes = []
        DFDataT = []
        DFDataG = []  # G is standard letter for conductance
        DFDataV = []
        counter = 0  # The counter is for downsampling
        for i, s in enumerate(self.states[rep]):
            if numpy.isnan(self.simDataTM[i]):  # reset counter with initialization (hold at pre-voltage)
                counter = downsample
            if counter >= downsample:  # Grab a data point
                counter = 0
                DFNodes.append(self.nodes[s])  # self.nodes are Node classes; s (in self.states) is an integer
                DFDataT.append(self.simDataTM[i])  # TM means Time Magnitude (no units)
                DFDataV.append(self.simDataVM[i])  # VM means Voltage Magnitude (no units)
                if hasG:
                    DFDataG.append(self.simDataGM[rep][i])
                else:
                    DFDataG.append(self.meansM[s])
            counter += 1
        TLabel = 'T_' + self.preferred.time
        VLabel = 'V_' + self.preferred.voltage
        GLabel = 'G_' + self.preferred.conductance
        dataDict = {TLabel: DFDataT, 'Node': DFNodes, VLabel: DFDataV, GLabel: DFDataG}
        return (pandas.DataFrame(dataDict, columns=[TLabel, 'Node', VLabel, GLabel]))

    def select(self, RNG, mat, row=0):  # select from matrix[row,:]
        # select is also defined in patch.singleChannelPatch
        p = RNG.random()
        rowsum = 0
        # cols should add to 1
        for col in range(mat.shape[1]):  # iterate over columns of mat
            rowsum += mat[row, col]  # row constant passed into select
            if p < rowsum:
                return col
        assert (False)  # Should never reach this point

    def makeB(self):  # Only good for no-noise
        self.B = []
        self.AB = []
        for uniqueLevel in range(len(self.levels)):
            # Blevel is the B-matrix for the observation of level==uniqueLevel
            Blevel = numpy.zeros([self.nStates, self.nStates])
            for d in range(self.nStates):  # Fill B with corresponding 1's
                if self.levelNum[d] == uniqueLevel:
                    Blevel[d, d] = 1
            self.B.append(Blevel)
            # ABlevel is AB-matricies for all voltage steps, at given level
            ABlevel = []
            # AVolt is A-matrix for given voltage
            for Avolt in self.A:
                ABlevel.append(Avolt.dot(Blevel))
            self.AB.append(ABlevel)

    def normalize(self, new):
        c = 1 / new.sum()
        return (c * new, c)

    def update(self, datum, distrib, k):
        new = distrib * self.B[datum[k]]  # n is traj num, k is sample num; B doesn't depend directly on voltage
        return self.normalize(new)

    def predictupdate(self, datum, distrib, k, iv):
        new = distrib * self.AB[datum[k]][iv]  # [[traj num][level num]][voltage num];  A depends on voltage
        return self.normalize(new)

    def likeOnce(self, datum):
        mll = 0.
        nextInitNum = 0
        k0 = 0
        for iv, ns in enumerate(self.nsamples):  # one nsample for each voltage step, equal number of samples in step
            if iv == 0 or ns == None:  # if nsamples == None then indicates an initialization at equilibrium distrib
                (alphak, ck) = self.update(datum, self.nextDistrib[nextInitNum], 0)  # don't pass in alphak
                mll += math.log(ck)
                nextInitNum += 1
                k0 += 1
                continue
            for k in range(k0,
                           k0 + ns):  # Next i (could follow intializatation or another voltage step without init)
                (alphak, ck) = self.predictupdate(datum, alphak, k, iv)  # pass in and return alphak
                mll += math.log(ck)
            k0 += ns
        return -mll
