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

#default_dt = parameter.Parameter("dt",0.05,"ms",log=True)
default_dt = parameter.Parameter("dt",5.,"ms",log=True)
default_tstop = parameter.Parameter("tstop",20.,"ms",log=True)

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
        self.setSampleInterval(default_dt)
    def setSampleInterval(self,dt):
        assert(parameter.v(dt)>0*u.milliseconds)  # dt > 0 regardless of units
        self.dt = dt
    def flatten(self,seed=None):
        parent = self # for readablility of pass to engine command
        FS = engine.flatStepProtocol(parent,seed)
        return FS

class singleChannelPatch(object):
    def __init__(self, ch):
        self.ch = ch
        self.Mean = self.ch.makeMean()
        self.Std = self.ch.makeStd()
        self.noise(False)
    def noise(self,toggle):
        self.hasNoise = toggle
        if self.hasNoise==False:
            self.ch.makeLevelMap()
            self.uniqueLevels = set(self.ch.uniqueLevels)
        else:
            assert(False)  # Take this out after implementing noise
    def getQ(self,volts):
        channel.VOLTAGE.remap(volts)
        return self.ch.makeQ()
    def getA(self,volts,dt):
        Q = self.getQ(volts)
        A = scipy.linalg.expm(dt*Q)
        # assert sum of rows is row of ones to tolerance
        tol = 1e-7
        assert(np.amin(np.sum(A,axis=1))>1.-tol)
        assert(np.amax(np.sum(A,axis=1))<1.+tol)
        return A
    def equilibrium(self,volts):
        return equilQ(self.getQ(volts))
    # Select is now also defined in engine.flatStepProtocol
    def select(self,R,mat,row=0):  # select from matrix[row,:]
        p = R.random()
        rowsum = 0
        # cols should add to 1
        for col in range(mat.shape[1]):  # iterate over columns of mat
            rowsum += mat[row, col]  # row constant passed into select
            if p < rowsum:
                return col
        assert(False) # Should never reach this point
