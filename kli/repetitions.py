__author__ = 'sean'
import numpy
import toy

class Repetitions(toy.FlatToy):
    def __init__(self, base, rReps):
        self.base = base
        self.rReps= rReps
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

    def extendBaseData(self, reps):
        mRepsBaseOriginal = self.base.mReps
        self.base.sim(reps)
        self.base.sim(mRepsBaseOriginal)

    def extendBaseLikes(self, reps, trueModel=None):
        if trueModel is None:
            trueModel = self
        mRepsBaseOriginal = trueModel.base.mReps
        trueModel.base.sim(reps)
        self.base.likelihoods(trueModel.base)
        trueModel.base.sim(mRepsBaseOriginal)

    def sim(self, mReps=None):
        if mReps is None:
            mReps = len(self.data)
        self.extendBaseData(self.rReps*mReps)
        for r in range(len(self.data),mReps):
            datum = [self.base.data[r*self.rReps + d] for d in range(self.rReps)]
            self.data.append(datum)
        self.mReps = mReps

    def likelihoods(self, trueModel=None):
        if trueModel is None:
            trueModel = self
        likes = trueModel.likes.getOrMakeEntry(self)
        baseLikes = trueModel.base.likes.getOrMakeEntry(self.base)
        self.extendBaseLikes(self.rReps*self.mReps, trueModel)
        nFirst = len(likes)
        nLast = trueModel.mReps
        for n in range(nFirst, nLast):
            likeum = [baseLikes[n*self.rReps + d] for d in range(self.rReps)]
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

    def mle(self):
        pass

    def lrN(self, alt, N, M):
        print "Don't call lrN from Reps class"
        assert False

    def aicN(self, alt, N, M):
        print "Don't call aicN from Reps class"
        assert False

