__author__ = 'sean'
import toy
import time
import numpy
import scipy.stats
import parameter


class NumSaveSeedRNG(numpy.random.RandomState):
    def __init__(self, seed=None):
        super(NumSaveSeedRNG, self).__init__()
        self.setSeedAndOffset(seed,0)
        self.setSeed(seed)

    def setSeedAndOffset(self, seed=None, offset=0):
        self.seedOffset = offset
        self.setSeed(seed)

    def setSeed(self, seed=None):
        if seed is None:
            self.usedSeed = long(time.time() * 256) + self.seedOffset
        else:
            self.usedSeed = [seed, self.seedOffset]
        self.reset()

    def reset(self):
        self.seed(self.usedSeed)


class Simple(object):
    def __init__(self, n, p, lam):
        self.n = n
        self.p = p
        self.lam = lam
        self.preferred = parameter.preferredUnits()
        self.preferred.freq = 'kHz'

    def flatten(self, seed=None):
        parent = self # for readability
        FS = FlatSimple(parent, seed)
        return FS

    def getExperiment(self):
        n = int(parameter.mu(self.n, 'dimensionless'))
        p = parameter.mu(self.p, 'dimensionless')
        lam = parameter.mu(self.lam, self.preferred.freq)
        return (n, p, lam)


class FlatSimple(toy.FlatToy):
    def initRNG(self, seed=None):
        return NumSaveSeedRNG(seed)

    def setUpExperiment(self, parent):
        self.experiment = parent.getExperiment()
        self.n, self.p, self.lam = self.experiment
        self.B = scipy.stats.binom(self.n, self.p)
        self.changedSinceLastSim = True

    def simulateOnce(self, RNG=None):
        if RNG is None:
            RNG = self.initRNG(None)
        return RNG.binomial(self.n, self.p)

    def likeOnce(self, datum):
        return self.B.logpmf(datum)

    def datumIntegrity(self, datum):
        if datum > self.n:
            return False
        else:
            return True
