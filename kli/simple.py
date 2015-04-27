__author__ = 'sean'
import toy
import parameter

class Simple(object):
    def __init__(self, p, lam):
        self.p = p
        self.lam = lam
        self.preferred = parameter.preferredUnits()
        self.preferred.freq = 'kHz'

    def flatten(self, seed=None):
        parent = self # for readability
        FS = FlatSimple(parent, seed)
        return FS

    def getExperiment(self):
        p = parameter.mu(self.p, 'dimensionless')
        lam = parameter.mu(self.lam, self.preferred.freq)