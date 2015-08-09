import toy

class TruncatedGaussian(object):
    def __init__(self, cv=.5, mu=1):
        self.cv = .5
        self.mu = 1

    def flatten(self, seed=None, name=None):
        parent = self  # for readability
        return FlatTruncatedGaussian(parent, seed, name)

    def getExperiment(self):
        return (self.cv, self.mu)


class FlatTruncatedGaussian(toy.FlatToy):
    def setUpExperiment(self, parent):
        self.experiment = parent.getExperiment()
        self.cv, self.mu = self.experiment
        self.sig = self.cv*self.mu

    def simulateOnce(self, RNG=None):
        if RNG is None:
            RNG = self.initRNG(None)
        x = -1.0
        while x < 0:
            x = RNG.normal(self.mu, self.sig)

    def likeOnce(self, datum):
        pass

    def datumWellFormed(self, datum):
        pass

    def datumIntegrity(self, datum):
        pass
