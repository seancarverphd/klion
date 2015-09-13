import numpy
import math
import scipy
import scipy.stats
import ad.admath
import toy

class TruncatedGaussian(object):
    def __init__(self, cv=.5, mu=1.):
        self.cv = cv
        self.mu = mu

    def flatten(self, seed=None, name=None):
        parent = self  # for readability
        return FlatTruncatedGaussian(parent, seed, name)

    def getExperiment(self):
        return {'cv': self.cv, 'mu': self.mu}

def TruncNormAlpha(mu_n,sig_n):
    return -mu_n/sig_n

def TruncNormLambda(mu_n,sig_n):   # use math module instead of numpy for automatic differentiation
    return ad.admath.exp(-mu_n**2./(2.*sig_n**2.)) \
        / (ad.admath.sqrt(2.*ad.admath.pi)*(1.-0.5*ad.admath.erfc(mu_n/(ad.admath.sqrt(2)*sig_n))))
        # / (ad.admath.sqrt(2.*ad.admath.pi)*sig_n*(1.-0.5*ad.admath.erfc(mu_n/(ad.admath.sqrt(2)*sig_n))))

def TruncNormDelta(mu_n,sig_n):
    return TruncNormLambda(mu_n,sig_n)*(TruncNormLambda(mu_n,sig_n)-TruncNormAlpha(mu_n,sig_n))

def TruncNormMean(mu_n,sig_n):
    return mu_n + sig_n*TruncNormLambda(mu_n,sig_n)

def TruncNormVariance(mu_n,sig_n):
    return sig_n**2*(1-TruncNormDelta(mu_n,sig_n))

# Both TruncNormMomentsErrorWikipediaFormula and TruncNormMomentsError seem to give same results (add unit tests)
def TruncNormMomentsErrorWikipediaFormula(mu_sig_notrunc, mu_sig_desired):
    mu_notrunc, sig_notrunc = mu_sig_notrunc # no truncation
    Norm = scipy.stats.norm(loc=mu_notrunc, scale=sig_notrunc)
    mu_desired, sig_desired = mu_sig_desired
    mu = mu_notrunc + Norm.pdf(0)*sig_notrunc/(1-Norm.cdf(0))
    var = sig_notrunc**2*(1 - (mu_notrunc/sig_notrunc)*Norm.pdf(0)/(1-Norm.cdf(0))
                            - (Norm.pdf(0)/(1-Norm.cdf(0)))**2)
    return (mu - mu_desired, var - sig_desired**2)

def TruncNormMomentsError(musig_n,mu_desired,sig_desired):
    mu_n, sig_n = musig_n # _n means no truncation
    return [TruncNormMean(mu_n, sig_n) - mu_desired, TruncNormVariance(mu_n, sig_n) - sig_desired**2]

def TruncNormMomentsErrorWithJacobian(musig_n, mu_desired, sig_desired):
    mu_n, sig_n = musig_n
    mu_nad = ad.adnumber(musig_n[0])
    sig_nad = ad.adnumber(musig_n[1])
    return (TruncNormMomentsError((mu_n, sig_n), mu_desired, sig_desired),
            ad.jacobian(TruncNormMomentsError((mu_nad, sig_nad),
                                              mu_desired, sig_desired), [mu_nad, sig_nad]))

def TruncNormMomentsErrorSciPy(mu_sig_notrunc, mu_desired, sig_desired):
    mu_notrunc, sig_notrunc = mu_sig_notrunc # no truncation
    a = -mu_notrunc/sig_notrunc
    b = mu_notrunc + 1000.*sig_notrunc  # b=infty causes problems
    TruncNorm = scipy.stats.truncnorm(a, b, loc=mu_notrunc, scale=sig_notrunc)
    return (TruncNorm.mean() - mu_desired, TruncNorm.var() - sig_desired**2)

class FlatTruncatedGaussian(toy.FlatToy):
    def unpackExperiment(self):
        self.cv = self.experiment['cv']
        self.mu = self.experiment['mu']
        self.sig = self.cv*self.mu
        self.mu_sig_norm = scipy.optimize.root(TruncNormMomentsErrorWithJacobian,
                                               (self.mu, self.sig), args=(self.mu, self.sig), jac=True)
        self.mu_norm, self.sig_norm = self.mu_sig_norm.x
        self.Norm = scipy.stats.norm(loc=self.mu_norm, scale=self.sig_norm)
        self.a = -self.mu_norm/self.sig_norm
        self.b = self.mu_norm + 1000.*self.sig_norm  # b=infty causes problems
        self.TruncNorm = scipy.stats.truncnorm(self.a, self.b, loc=self.mu_norm, scale=self.sig_norm)

    def simulateOnce(self, RNG=None):
        if RNG is None:
            RNG = self.initRNG(None)
        x = -1.
        i = 0
        while x < 0.:
            assert i < 100
            i += 1
            x = RNG.normal(self.mu_norm, self.sig_norm)
            # not too inefficient because for most parameter values we care about, x will usually be positive
        return x

    def likeOnce(self, datum):
        if datum < 0:
            return -numpy.infty
        else:
            return numpy.log(self.Norm.pdf(datum)/(1.-self.Norm.cdf(0)))

    def datumWellFormed(self, datum):
        return isinstance(numpy.pi, float)

    def datumIntegrity(self, datum):
        return self.datumWellFormed(datum) and (datum >= 0)

class InverseGaussian(TruncatedGaussian):
    def flatten(self, seed=None, name=None):
        parent = self  # for readability
        return FlatInverseGaussian(parent, seed, name)

class FlatInverseGaussian(toy.FlatToy):
    def unpackExperiment(self):
        self.cv = self.experiment['cv']
        self.mu = self.experiment['mu']
        self.shape = self.mu/(self.cv**2)
        self.sig = self.mu*self.cv
        # Removed following line because it's buggy (at least in Scipy Version 0.13.3)
        # self.IG = scipy.stats.invgauss(mu=self.mu, scale=self.scale)

    def simulateOnce(self, RNG=None):
        if RNG is None:
            RNG = self.initRNG(None)
        return RNG.wald(self.mu, self.shape)

    def likeOnce(self, datum):
        # Using Wikipedia Formula because the function in Scipy Version 0.13.3 gives erroneous results!
        if datum < 0:
            return -numpy.infty
        else:
            return .5*(numpy.log(self.shape) - numpy.log(2.*numpy.pi) - 3.*numpy.log(datum)) \
                    + (-self.shape*(datum - self.mu)**2/(2.*(self.mu**2)*datum))

    def datumWellFormed(self, datum):
        return isinstance(numpy.pi, float)

    def datumIntegrity(self, datum):
        return self.datumWellFormed(datum) and (datum >= 0)

if __name__ == '__main__':
    TG = TruncatedGaussian(cv=.5)
    FTG = TG.flatten(name='Truncated Gaussian: CV=0.5')
    IG = InverseGaussian(cv=.5)
    FIG = IG.flatten(name='Inverse Gaussian: CV=0.5')
    TG10 = TruncatedGaussian(cv=.1)
    FTG10 = TG10.flatten(name='Truncated Gaussian: CV=0.1')
    IG10 = InverseGaussian(cv=.1)
    FIG10 = IG10.flatten(name='Inverse Gaussian: CV=0.1')
    FIG.sim(1000)
    FTG.sim(1000)
    FIG10.sim(1000)
    FTG10.sim(1000)
    FIG.compare_pdfs(FTG,a=0.001,b=6,xlab='Intertick Interval Normalized to Multiples of Period')
    FIG10.compare_pdfs(FTG10,a=0.001,b=6,xlab='Intertick Interval Normalized to Multiples of Period')