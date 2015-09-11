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
import engine

# default_dt = parameter.Parameter("dt",0.05,"ms",log=True)
default_dt = parameter.Parameter("dt", 0.01, "ms", log=True)
default_tstop = parameter.Parameter("tstop", 20., "ms", log=True)

preferred = parameter.preferredUnits()
preferred.time = 'ms'
preferred.voltage = 'mV'
preferred.conductance = 'pS'
preferred.current = "fA"

class StepProtocol(object):
    def __init__(self, patch, voltages, voltageStepDurations):
        self.thePatch = patch
        self.voltages = voltages
        self.voltageStepDurations = voltageStepDurations
        self.setSampleInterval(default_dt)
        self.preferred = preferred

    def setSampleInterval(self, dt):
        assert (parameter.v(dt) > 0 * u.milliseconds)  # dt > 0 regardless of units
        self.dt = dt

    def flatten(self, seed=None):
        parent = self  # for readablility of pass to engine command
        FS = engine.FlatStepProtocol(parent, seed)
        return FS

    def getExperiment(self):
        return {'hasNoise': self.thePatch.hasNoise,  # Later will implement NOISE
                'preferredTime': self.preferred.time,  # preferred time unit
                'preferredVoltage': self.preferred.voltage,  # preferred voltage unit
                'preferredConductance' : self.preferred.conductance, # preferred conductance unit
                'dt': parameter.mu(self.dt, self.preferred.time),  # self.dt a number
                'voltages': tuple([parameter.mu(v, self.preferred.voltage)
                                  for v in self.voltages]),
                'durations': tuple([parameter.mu(dur, self.preferred.time)
                           for dur in self.voltageStepDurations]),
                'thePatch': self.thePatch}

class singleChannelPatch(object):
    def __init__(self, ch, VOLTAGE):
        self.ch = ch
        self.VOLTAGE = VOLTAGE
        self.Mean = self.ch.makeMean()
        self.Std = self.ch.makeStd()
        self.noise(False)

    def noise(self, toggle):
        self.hasNoise = toggle
        if self.hasNoise == False:
            self.ch.makeLevelMap()
            self.uniqueLevels = set(self.ch.uniqueLevels)
        else:
            assert (False)  # Take this out after implementing noise

    def makeQ(self, volts, voltageUnit=None):
        if voltageUnit is not None:
            volts = volts*parameter.u.__getattr__(voltageUnit)
        self.VOLTAGE.remap(volts)
        return self.ch.makeQ()

    def makeA(self, volts, dt, voltageUnit=None, timeUnit=None):
        Q = self.makeQ(volts, voltageUnit)
        if timeUnit is not None:
            dt = dt * parameter.u.__getattr__(timeUnit)
        A = scipy.linalg.expm(dt * Q)
        self.assertSumOfRowsIsRowOfOnes(A)
        return A

    def assertSumOfRowsIsRowOfOnes(self,A):
        # assert sum of rows is row of ones to tolerance
        tol = 1e-7
        assert (np.amin(np.sum(A, axis=1)) > 1. - tol)
        assert (np.amax(np.sum(A, axis=1)) < 1. + tol)

    def equilibrium(self, volts, voltageUnit=None, timeUnit=None):
        Qunits = self.makeQ(volts, voltageUnit)
        Q = parameter.mu(Qunits, '1/'+timeUnit)
        (V, D) = np.linalg.eig(Q.T)  # eigenspace
        imin = np.argmin(np.absolute(V))  # index of 0 eigenvalue
        eigvect0 = D[:, imin]  # corresponding eigenvector
        return eigvect0.T / sum(eigvect0)  # normalize and return (fixes sign)

    # Select is now also defined in engine.flatStepProtocol
    def select(self, R, mat, row=0):  # select from matrix[row,:]
        p = R.random()
        rowsum = 0
        # cols should add to 1
        for col in range(mat.shape[1]):  # iterate over columns of mat
            rowsum += mat[row, col]  # row constant passed into select
            if p < rowsum:
                return col
        assert False  # Should never reach this point


khhPatch = singleChannelPatch(channel.khh, channel.VOLTAGE)
SP = StepProtocol(khhPatch, [-65*u.mV, -20*u.mV], [np.inf, 10*u.ms])
FS = SP.flatten(5)
FS.sim(10)
testedlike = FS.like()
