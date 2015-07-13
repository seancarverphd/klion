__author__ = 'sean'
import numpy
import toy
import scipy

class Repetitions(toy.FlatToy):
    def __init__(self, base, rReps):
        self.base = base
        self.rReps= rReps
        self.stack = []
        super(Repetitions, self).__init__(base)

    def defineRepetitions(self):
        pass

    def initRNG(self, seed=None):
        pass

    def setUpExperiment(self, base):
        pass
        # self.base = base # moved to __init__() to avoid errors in debug

    def _reseed(self, seed=None):
        self.base._reseed(seed)

    def _restart(self):
        self.data = []
        self.likes = []
        self.base._restart()

    def push_mReps(self, reps=None):
        if reps is None:
            reps = self.mReps * self.rReps
        self.stack.append(self.base.mReps)
        self.base.sim(reps)

    def pop_mReps(self):
        assert len(self.stack) > 0
        reps = self.stack.pop(-1)
        self.base.sim(reps)

    def extendBaseData(self, reps):
        self.push_mReps(reps)
        self.pop_mReps()
        # mRepsBaseOriginal = self.base.mReps
        # self.base.sim(reps)
        # self.base.sim(mRepsBaseOriginal)

    def extendBaseLikes(self, trueModel=None, reps=None):
        if trueModel is None:
            trueModel = self
        # if reps is None:
        #     reps = trueModel.rReps * trueModel.mReps
        # mRepsBaseOriginal = trueModel.base.mReps
        # trueModel.base.sim(reps)
        trueModel.push_mReps(reps)
        self.base.likelihoods(trueModel.base)
        trueModel.pop_mReps()
        # trueModel.base.sim(mRepsBaseOriginal)

    def sim(self, mReps=None):
        if mReps is None:
            mReps = int(len(self.base.data) / self.rReps)
        self.extendBaseData(self.rReps*mReps)
        for r in range(len(self.data),mReps):
            datum = [self.base.data[r*self.rReps + d] for d in range(self.rReps)]
            self.data.append(datum)
        self.mReps = mReps

    def likelihoods(self, trueModel=None):
        if trueModel is None:
            trueModel = self
        assert trueModel.rReps == self.rReps
        likes = trueModel.likes.getOrMakeEntry(self)
        baseLikes = trueModel.base.likes.getOrMakeEntry(self.base)
        self.extendBaseLikes(trueModel)
        nFirst = len(likes)
        nLast = trueModel.mReps
        for n in range(nFirst, nLast):
            likeum = [baseLikes[n*trueModel.rReps + d] for d in range(trueModel.rReps)]
            likes.append(sum(likeum))
        return likes[0:nLast]

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

    def PFalsifyNormal(self, alt, trueModel=None):
        if trueModel is None:
            trueModel = self
        trueModel.push_mReps()
        # mRepsBaseOriginal = trueModel.base.mReps
        # trueModel.base.sim(trueModel.mReps * trueModel.rReps)
        mu, sig = self.base.likeRatioMuSigma(alt.base, trueModel.base)
        trueModel.pop_mReps()
        # trueModel.base.sim(mRepsBaseOriginal)
        return scipy.stats.norm.cdf(numpy.sqrt(self.rReps)*mu/sig)

    def rInfinity(self, alt, trueModel=None, C=0.95):
        if trueModel is None:
            trueModel = self
        trueModel.push_mReps()
        mu, sig = self.base.likeRatioMuSigma(alt.base, trueModel.base)
        trueModel.pop_mReps()
        return (scipy.stats.norm.ppf(C)*sig/mu)**2

    def rPlus(self, alt, trueModel=None, rMinus=None, PrMinus=None, C=0.95):
        if trueModel is None:
            trueModel = self
        if rMinus is None:
            rMinus = self.rInfinity(alt, trueModel, C)
        if PrMinus is None:
            newReps = max(1, int(rMinus))
            repeated_self = Repetitions(self.base, newReps)
            repeated_alt = Repetitions(alt.base, newReps)
            repeated_true = Repetitions(trueModel.base, newReps)
            repeated_true.sim()
            PrMinus = repeated_self.PFalsify(repeated_alt, repeated_true)
        cv = numpy.sqrt(rMinus)/scipy.stats.norm.ppf(PrMinus)
        return (scipy.stats.norm.ppf(C)*cv)**2

    def rStar(self, alt, trueModel=None, rMinus=None, PrMinus=None, C=0.95):
        pass

    def lrN(self, alt, N, M):
        print "Don't call lrN from Reps class"
        assert False

    def aicN(self, alt, N, M):
        print "Don't call aicN from Reps class"
        assert False

