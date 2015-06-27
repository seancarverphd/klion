import numpy
import math
import random
import time
import copy
import parameter
import pandas
import toy
from parameter import u

class flatStepProtocol(toy.FlatToy):
    def initRNG(self, seed):
        return toy.MultipleRNGs(2,seed) # Two instances of random.Random with seed save added

    def _restart(self):  # Clears data and resets RNG with same seed
        super(flatStepProtocol, self)._restart()
        self.simDataGM = []  # conductance, simulated separately.

    def setUpExperiment(self, parent):
        assert not parent.thePatch.hasNoise  # Later will implement NOISE
        self.preferredTime = parent.preferred.time  # preferred time unit
        self.preferredVoltage = parent.preferred.voltage # preferred voltage unit
        self.preferredConductance = parent.preferred.conductance # preferred conductance unit

        self.dt = parameter.mu(parent.dt, self.preferredTime)  # self.dt a number
        self.voltages = tuple([parameter.mu(v, self.preferredVoltage)
                               for v in parent.voltages])
        self.durations = tuple([parameter.mu(dur, self.preferredTime)
                                for dur in parent.voltageStepDurations])
        self.nsamples = tuple([None if numpy.isinf(dur) else int(dur/self.dt)
                                for dur in self.durations])
        self.allInitializations = self.setUpInitializations(parent.thePatch.ch.timeZeroDistribution(),
                parent.thePatch.equilibrium)  # equilibrium is a function
        self.processNodes(parent.thePatch.ch.nodes)
        self.A = tuple([parent.thePatch.makeA(v, self.dt,
                                            self.preferredVoltage,
                                            self.preferredTime) for v in self.voltages])
        self.makeB()  # NO-NOISE only.
        self.changedSinceLastSim = True
        self.hasVoltTraj = False  # hasVoltTraj used in self.voltageTrajectory() for dataFrame

    def _changeModel(self, parent, integrityCheck=True,
                    nodesChanged=True, QChanged=True):
        if integrityCheck:
            assert not parent.thePatch.hasNoise  # Later will implement NOISE
            assert self.preferredTime == parent.preferred.time  # preferred time unit
            assert self.preferredVoltage == parent.preferred.voltage # preferred voltage unit
            assert self.preferredConductance == parent.preferred.conductance # preferred conductance unit
            assert self.dt == parameter.mu(parent.dt, self.preferredTime)  # self.dt a number
            assert self.voltages == tuple([parameter.mu(v, self.preferredVoltage)
                                for v in parent.voltages])
            assert self.durations == tuple([parameter.mu(dur, self.preferredTime)
                                for dur in parent.voltageStepDurations])
            assert self.nsamples == tuple([None if numpy.isinf(dur) else int(dur/self.dt)
                                for dur in self.durations])
            assert (parent.thePatch.ch.timeZeroDistribution() is None or
                    parent.thePatch.ch.timeZeroDistribution() == self.allInitializations[0])
            # For No Noise Must Have Same Level
            assert set(self.levelNames) == {str(n.level)
                                            for n in parent.thePatch.ch.nodes}  # list(SET) makes unique
        # if self.NoiseChanged:
        #     self.means == tuple([parameter.mu(n.level.mean,
        #                         self.preferredConductance) for n in parent.thePatch.ch.nodes])
        #     self.stds = tuple([parameter.mu(n.level.std,
        #                        self.preferredConductance) for n in parent.thePatch.ch.nodes])
        if nodesChanged:
            self.processNodes(parent.thePatch.ch.nodes)
        if QChanged:
            self.A = tuple([parent.thePatch.makeA(v, self.dt,
                                            self.preferredVoltage,
                                            self.preferredTime) for v in self.voltages])
            self.makeB()  # NO-NOISE only.
        self.changedSinceLastSim = True

    def setUpInitializations(self, timeZeroInitialization, equilibrium):
        # Initializations occur when the voltage clamp is held for a long time without collecting
        # data. The initializations, except possibly the first one at time zero are determined
        # by the equilibrium distribution at the holding potential for the initialization.
        allInitializations = []
        if timeZeroInitialization is None:  # No initial distribution because all weights are zero,
                                            # use equilibrium distribution same as other initializations
            assert (self.nsamples[0] is None)  # Then must have infinite duration for first voltage,
                                               # to use use equilibrium distribution
        else:
            assert (self.nsamples[0] is not None)  # Finite duration for first voltage. Use timeZeroInitialization
            allInitializations.append(timeZeroInitialization)  # Use timeZeroInitialization
                                                                    # for initial distribution
        # Now we generate an initialization distribution where nsamples is None
        for i, ns in enumerate(self.nsamples):
            if ns is None:  # Requires new initialization of state when simulating
                allInitializations.append(equilibrium(self.voltages[i], self.preferredVoltage, self.preferredTime))
        return tuple(allInitializations)

    def processNodes(self, nodes):
        self.nStates = len(nodes)
        self.nodeNames = tuple([str(n) for n in nodes])
        self.levelNames = tuple({str(n.level) for n in nodes})  # list(SET) makes unique
        self.levelMap = tuple([str(n.level) for n in nodes])
        self.means = tuple([parameter.mu(n.level.mean,
                                   self.preferredConductance) for n in nodes])
        self.stds = tuple([parameter.mu(n.level.std,
                                  self.preferredConductance) for n in nodes])

    def makeB(self):  # Only good for no-noise
        self.B = {}
        self.AB = {}
        for levelName in self.levelNames:
            # Blevel is the B-matrix for the observation of level==uniqueLevel
            Blevel = numpy.zeros([self.nStates, self.nStates])
            for d in range(self.nStates):  # Fill B with corresponding 1's
                if self.levelMap[d] == levelName:
                    Blevel[d, d] = 1
            self.B.update({levelName: Blevel})
            # ABlevel is AB-matrices for all voltage steps, at given level
            ABlevel = []
            # AVolt is A-matrix for given voltage
            for Avolt in self.A:  # self.A is a list of A matrix, one for each voltage
                ABlevel.append(Avolt.dot(Blevel))
            self.AB.update({levelName: ABlevel}) # Dictionary of AB lists over voltage

    def simulateOnce(self, RNG=None):
        if RNG is None:
            RNG = self.initRNG(None)
        self.hiddenStateTrajectory = []
        levelsTrajectory = []
        nextInitNum = 0
        for i, ns in enumerate(self.nsamples):  # one nsample for each voltage step, equal number of samples in step
            if i == 0 or ns == None:  # if nsamples == None then indicates an initialization at equilibrium distrib
                state = self.nextInit(RNG.RNGs[0], nextInitNum)  # Next: append state and level to simS and simL
                self.appendTrajectory(state, self.hiddenStateTrajectory, levelsTrajectory)  # refs to appendTrajectory
                nextInitNum += 1
                continue
            for j in range(ns):  # Next i (could follow intializatation or another voltage step without init)
                state = self.select(RNG.RNGs[0], self.A[i], state)
                self.appendTrajectory(state, self.hiddenStateTrajectory, levelsTrajectory)  # Pass ref to simS & simL so that appendTrajectory works
        return levelsTrajectory

    def nextInit(self, RNG, nextInitNum):  # initializes state based on stored equilibrium distributions
        return self.select(RNG, self.allInitializations[nextInitNum])

    def appendTrajectory(self, state, saveStates, saveLevels):
        if self.debugFlag:
            saveStates.append(self.nodeNames[state])
        # NO NOISE:
        saveLevels.append(self.levelMap[state])  # use self.levelMap for actual levels (not nums)
        # NOISE:
        # Might want to modify next line: multiply conductance by "voltage" to get current
        # where I think "voltage" should really be difference between voltage and reversal potential
        # self.simDataX.append(self.R.normalvariate(self.Mean[state],self.Std[state]))
        # self.simDataX.append(self.Mean[state])
        # IF SAVING VOLTAGE:
        # self.simDataV.append(volts)

    def select(self, RNG, mat, row=0):  # select from matrix[row,:]
        # select is also defined in patch.singleChannelPatch
        p = RNG.random()
        rowsum = 0
        # cols should add to 1
        for col in range(mat.shape[1]):  # iterate over columns of mat
            rowsum += mat[row, col]  # row constant passed into select
            if p < rowsum:
                 return col
        assert False  # Should never reach this point

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
        if self.debugFlag:
            self.recentLikeInfo = []
        for iv, ns in enumerate(self.nsamples):  # one nsample for each voltage step, equal number of samples in step
            if iv == 0 or ns == None:  # if nsamples == None then indicates an initialization at equilibrium distrib
                (alphak, ck) = self.update(datum, self.allInitializations[nextInitNum], 0)  # don't pass in alphak
                mll += math.log(ck)
                nextInitNum += 1
                k0 += 1
                if self.debugFlag:
                    self.recentLikeInfo.append((alphak, mll))
                continue
            for k in range(k0,
                           k0 + ns):  # Next i (could follow intializatation or another voltage step without init)
                (alphak, ck) = self.predictupdate(datum, alphak, k, iv)  # pass in and return alphak
                mll += math.log(ck)
                if self.debugFlag:
                    self.recentLikeInfo.append((alphak, mll))
            k0 += ns
        return -mll

    def simDataFrame(self, rep=0, downsample=0):
        self.voltageTrajectory()
        DFNodes = []
        DFDataT = []
        DFDataV = []
        counter = 0  # The counter is for downsampling
        for i, s in enumerate(self.hiddenStates[rep]):
            if numpy.isnan(self.simDataTM[i]):  # reset counter with initialization (hold at pre-voltage)
                counter = downsample
            if counter >= downsample:  # Grab a data point
                counter = 0
                DFNodes.append(s)  # s is an integer
                DFDataT.append(self.simDataTM[i])  # TM means Time Magnitude (no units)
                DFDataV.append(self.simDataVM[i])  # VM means Voltage Magnitude (no units)
            counter += 1
        TLabel = 'T_' + self.preferredTime
        VLabel = 'V_' + self.preferredVoltage
        dataDict = {TLabel: DFDataT, 'Node': DFNodes, VLabel: DFDataV}
        return (pandas.DataFrame(dataDict, columns=[TLabel, 'Node', VLabel]))

    def likeDataFrame(self,rep=0, downsample=0):
        PC0 = []
        PC1 = []
        POpen = []
        Lmll = []
        for i, sample in enumerate(self.data[rep]):
            PC0.append(self.likeInfo[rep][i][0][0,0])
            PC1.append(self.likeInfo[rep][i][0][0,1])
            POpen.append(self.likeInfo[rep][i][0][0,2])
            Lmll.append(self.likeInfo[rep][i][1])
        likeDict = {'PC0': PC0, 'PC1': PC1, 'POpen': POpen, 'Lmll': Lmll}
        return pandas.DataFrame(likeDict)

    def voltageTrajectory(self):
        """Compute the trajectory of the holding voltage as a function of time"""
        # The voltageTrajectory only depends on the Protocol not model.
        if self.hasVoltTraj:  # changeProtocol sets this to False
            return
        # self.voltagesM = []  # Strip units off voltages
        # for v in self.voltages:
        #     self.voltagesM.append(parameter.mu(v, self.preferredVoltage))
        self.simDataVM = []
        self.simDataTM = []
        # self.simDataV.append(self.voltages[0])
        for i, ns in enumerate(self.nsamples):  # one nsample per voltage, so iterates over voltages
            if ns == None:
                timeM = 0  # no units, M is for magnitude (no units)
                self.simDataTM.append(numpy.nan)
                self.simDataVM.append(self.voltages[i])
                continue
            elif i == 0:  # not ns==None and i==0
                timeM = 0  # no units
                self.simDataTM.append(timeM)
                self.simDataVM.append(numpy.nan)
            for j in range(ns):
                timeM += self.dt
                self.simDataTM.append(timeM)
                self.simDataVM.append(self.voltages[i])  # same voltage every sample until voltage steps
        self.hasVoltTraj = True
