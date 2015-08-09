import numpy
import scipy
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
        self.Norm = scipy.stats.norm(loc=self.mu,scale=self.sig)

    def simulateOnce(self, RNG=None):
        if RNG is None:
            RNG = self.initRNG(None)
        x = -1.0
        while x < 0:
            x = RNG.normal(self.mu, self.sig)
        return x

    def likeOnce(self, datum):
        if datum < 0:
            return 0
        else:
            return self.Norm.pdf(datum)/(1.-self.Norm.cdf(0))

    def datumWellFormed(self, datum):
        return isinstance(numpy.pi, float)

    def datumIntegrity(self, datum):
        return self.datumWellFormed(datum) and (datum >= 0)

class InverseGaussian(TruncatedGaussian):
    def flatten(self, seed=None, name=None):
        parent = self  # for readability
        return FlatInverseGaussian(parent, seed, name)

class FlatInverseGaussian(toy.FlatToy):
    def setUpExperiment(self, parent):
        self.experiment = parent.getExperiment()
        self.cv, self.mu = self.experiment
        self.scale = self.mu/(self.cv**2)
        self.IG = scipy.stats.invgauss(mu=self.mu, scale=self.scale)

    def simulateOnce(self, RNG=None):
        if RNG is None:
            RNG = self.initRNG(None)
        x = -1.0
        while x < 0:
            x = RNG.wald(self.mu, self.scale)
        return x

    def likeOnce(self, datum):
        if datum < 0:
            return 0
        else:
            return self.IG.pdf(datum)

    def datumWellFormed(self, datum):
        return isinstance(numpy.pi, float)

    def datumIntegrity(self, datum):
        return self.datumWellFormed(datum) and (datum >= 0)

if __name__ == '__main__':
    TG = TruncatedGaussian(cv=.5)
    FTG = TG.flatten(name='FTG')
    IG = InverseGaussian(cv=.5)
    FIG = IG.flatten(name='FIG')
