__author__ = 'sean'
import numpy
import toy
import scipy

class Repetitions(toy.FlatToy):
    def __init__(self, base, rReps, name=None):
        self.base = base
        self.rReps= rReps
        self.stack = []
        super(Repetitions, self).__init__(base, seed=None, name=name)

    def str_reps(self):
        return '(%d reps of %s)' % (self.rReps, self.base.str_name())

    def __str__(self):
        first = super(Repetitions, self).__str__()
        last = self.str_reps()
        if len(first)+len(last) > 40:
            middle = '\n'
        else:
            middle = ' '
        return first + middle +  last

    def defineRepetitions(self):
        pass

    def setUpExperiment(self, base):
        pass

    def _reseed(self, seed=None):
        self.base._reseed(seed)

    def _restart(self):
        self.data = []
        self.likes = []
        self.base._restart()

    def set_base_mReps_to_mr(self, m=None, r=None):
        if m is None:
            m = self.mReps
        if r is None:
            r = self.rReps
        self.stack.append(self.base.mReps)
        self.base.sim(m*r)

    def pop_base_mReps(self):
        assert len(self.stack) > 0
        mReps = self.stack.pop(-1)
        self.base.sim(mReps)

    def extendBaseData(self, m=None, r=None):
        self.set_base_mReps_to_mr(m, r)
        self.pop_base_mReps()

    def extendBaseLikes(self, trueModel=None, m=None, r=None):
        if trueModel is None:
            trueModel = self
        trueModel.set_base_mReps_to_mr(m, r)
        self.base.likelihoods_monte_carlo_sample(trueModel.base)
        trueModel.pop_base_mReps()
        return trueModel.base.likes.getOrMakeEntry(self.base)

    def sim(self, mReps=None):
        if mReps is None:
            mReps = int(len(self.base.data) / self.rReps)
        self.mReps = mReps
        self.extendBaseData()
        for r in range(len(self.data), mReps):
            datum = [self.base.data[r*self.rReps + d] for d in range(self.rReps)]
            self.data.append(datum)

    def likelihoods_construct_from_base(self, trueModel=None, bootstrap=False):
        if trueModel is None:
            trueModel = self
        assert trueModel.rReps == self.rReps
        likes = []
        baseLikes = self.extendBaseLikes(trueModel)  # returns trueModel.base.likes
        if bootstrap:
            baseLikes = trueModel.resample(baseLikes, trueModel.bReps*trueModel.rReps)
            nLast = trueModel.bReps
        else:
            nLast = trueModel.mReps
        for n in range(nLast):
            like = [baseLikes[n*trueModel.rReps + d] for d in range(trueModel.rReps)]
            likes.append(sum(like))
        if self.debugFlag:
            print "likes = ", likes
        return likes

    def likelihoods_monte_carlo_sample(self, trueModel=None):
        return self.likelihoods_construct_from_base(trueModel, bootstrap=False)

    def likelihoods_bootstrap_sample(self, trueModel=None):
        return self.likelihoods_construct_from_base(trueModel, bootstrap=True)

    def _debug(self, flag=None):
        return self.base._debug(flag)

    def simulateOnce(self, RNG=None):
        return [self.base.simulateOnce(RNG) for d in range(self.rReps)]

    # Bad because recomputes likes rather than using base.likes
    def likeOnce(self, datum):
        assert self.datumWellFormed(datum)
        logLike = 0
        for datumComponent in datum:
            logLike += self.base.likeOnce(datumComponent)  # These are numbers
        return logLike

    def datumWellFormed(self, datum):
        mustBeTrue = (len(datum) == self.rReps)
        for d in datum:
            mustBeTrue = (mustBeTrue and self.base.datumWellFormed(d))
        return mustBeTrue

    def PFalsify_function_of_rReps(self, alt, trueModel=None,
                                   rRepsList=True, mReps=True, bReps=True,
                                   bootstrap_seed=None):
        if trueModel is None:
            trueModel = self
        if rRepsList is True:  # use trueModel (singleton list, default)
            rRepsList = [trueModel.rReps]  # Can pass longer list: e.g rRepsList=[1, 2, 3, 4, 5, 6]
        if mReps is True:  # use trueModel (default)
            mReps = trueModel.mReps
        if bReps is True:  # use trueModel (default)
            bReps = trueModel.bReps
        self.extendBaseLikes(trueModel, mReps, max(rRepsList))
        local_bootstrap_RNG = self.initRNG(bootstrap_seed)
        probabilities = []
        for r in rRepsList:
            repeated_self, repeated_alt, repeated_true = self.repeated_models(alt, trueModel, r, mReps)
            repeated_true.bootstrap(bReps, RNG=local_bootstrap_RNG)
            Pr = repeated_self.PFalsify(repeated_alt, repeated_true)
            probabilities.append(Pr)
        return probabilities

    def PFalsifyNormal(self, alt, trueModel=None):
        if trueModel is None:
            trueModel = self
        trueModel.set_base_mReps_to_mr()
        mu, sig = self.base.likeRatioMuSigma(alt.base, trueModel.base)
        trueModel.pop_base_mReps()
        return scipy.stats.norm.cdf(numpy.sqrt(self.rReps)*mu/sig)

    def rInfinity(self, alt, trueModel=None, C=0.95):
        if trueModel is None:
            trueModel = self
        trueModel.set_base_mReps_to_mr()
        mu, sig = self.base.likeRatioMuSigma(alt.base, trueModel.base)
        trueModel.pop_base_mReps()
        return (scipy.stats.norm.ppf(C)*sig/mu)**2

    def repeated_models(self, alt, trueModel=None, rReps=1, mReps=1):
        if trueModel is None:
            trueModel = self
        repeated_self = Repetitions(self.base, rReps)
        repeated_alt = Repetitions(alt.base, rReps)
        if trueModel is self:
            repeated_true = repeated_self
        elif trueModel is alt:
            repeated_true = repeated_alt
        else:
            repeated_true = Repetitions(trueModel.base, rReps)
        repeated_true.sim(mReps)  # if reps is None then use available data in trueModel.base
        return repeated_self, repeated_alt, repeated_true

    def rPlus(self, alt, trueModel=None, rMinus=None, PrMinus=None, C=0.95, mReps=None):
        if trueModel is None:
            trueModel = self
        if rMinus is None:
            rMinus = self.rInfinity(alt, trueModel, C)
        rMinus = max(1, int(rMinus))  # Make a positive integer
        if PrMinus is None:
            repeated_self, repeated_alt, repeated_true = self.repeated_models(alt, trueModel, rMinus, mReps)
            PrMinus = repeated_self.PFalsify(repeated_alt, repeated_true)
        cv = numpy.sqrt(rMinus)/scipy.stats.norm.ppf(PrMinus)
        return (scipy.stats.norm.ppf(C)*cv)**2

    def rMinus2Plus_plot(self, alt, trueModel, rMinus, rPlus, C):
        pass

    def rStar(self, alt, trueModel=None, rMinus=None, C=0.95, reps=None, iter=10, plot=False):
        for i in range(iter):
            rMinus = rPlus if i > 0 else rMinus
            rPlus = self.rPlus(alt, trueModel, rMinus, None, C, reps)
            print "Iteration: ", i, "| Value of R:", rMinus
        if plot:
            self.rMinus2Plus_plot(alt, trueModel, rMinus, rPlus, C)
        return rMinus

    def lrN(self, alt, N, M):
        print "Don't call lrN from Reps class"
        assert False

    def aicN(self, alt, N, M):
        print "Don't call aicN from Reps class"
        assert False

