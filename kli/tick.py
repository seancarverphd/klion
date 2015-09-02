import numpy
import scipy
import scipy.stats
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

# Both TruncNormMomentsErrorWikipediaFormula and TruncNormMomentsError seem to give same results (add unit tests)
def TruncNormMomentsErrorWikipediaFormula(mu_sig_notrunc, mu_sig_guess):
    mu_notrunc, sig_notrunc = mu_sig_notrunc # no truncation
    Norm = scipy.stats.norm(loc=mu_notrunc, scale=sig_notrunc)
    mu_guess, sig_guess = mu_sig_guess
    mu = mu_notrunc + Norm.pdf(0)*sig_notrunc/(1-Norm.cdf(0))
    var = sig_notrunc**2*(1 - (mu_notrunc/sig_notrunc)*Norm.pdf(0)/(1-Norm.cdf(0))
                            - (Norm.pdf(0)/(1-Norm.cdf(0)))**2)
    return (mu - mu_guess, var - sig_guess**2)

def TruncNormMomentsError(mu_sig_notrunc, mu_sig_guess):
    mu_notrunc, sig_notrunc = mu_sig_notrunc # no truncation
    a = -mu_notrunc/sig_notrunc
    b = mu_notrunc + 1000.*sig_notrunc  # b=infty causes problems
    TruncNorm = scipy.stats.truncnorm(a, b, loc=mu_notrunc, scale=sig_notrunc)
    mu_guess, sig_guess = mu_sig_guess
    return (TruncNorm.mean() - mu_guess, TruncNorm.var() - sig_guess**2)

class FlatTruncatedGaussian(toy.FlatToy):
    def setUpExperiment(self, parent):
        self.experiment = parent.getExperiment()
        self.cv, self.mu = self.experiment
        self.sig = self.cv*self.mu
        self.Norm = scipy.stats.norm(loc=self.mu,scale=self.sig)

    def simulateOnce(self, RNG=None):
        if RNG is None:
            RNG = self.initRNG(None)
        x = -1.
        while x < 0.:
            x = RNG.normal(self.mu, self.sig)
            # not too inefficient because for most parameter values we care about, x will usually be positive
        return x

    def likeOnce(self, datum):
        if datum < 0:
            return -numpy.infty
        else:
            return self.Norm.logpdf(datum)/(1.-self.Norm.cdf(0))

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
        return self.IG.logpdf(datum)

    def datumWellFormed(self, datum):
        return isinstance(numpy.pi, float)

    def datumIntegrity(self, datum):
        return self.datumWellFormed(datum) and (datum >= 0)

if __name__ == '__main__':
    TG = TruncatedGaussian(cv=.5)
    FTG = TG.flatten(name='FTG')
    IG = InverseGaussian(cv=.5)
    FIG = IG.flatten(name='FIG')
