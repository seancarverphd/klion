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
        # No changedSinceLastSim, don't want to clear base

    def _reseed(self, seed=None):
        self.base._reseed(seed)

    def _restart(self):
        self.data = []
        self.likes = []
        self.base._restart()

    def sim(self, mReps=1, clear=False):
        if clear:
            self._restart()
        # No changedSinceLastSim, don't want to clear base unless explicitly directed
        numOldReps = len(self.data)
        numNewReps = mReps - len(self.data)
        self.base.sim(self.rReps*mReps)  # Simulate base and only compute newly needed data
        for r in range(numNewReps):
            datum = []
            for d in range(self.rReps):
                datum.append(self.base.data[(numOldReps+r)*self.rReps + d])
            self.data.append(datum)
        self.mReps = mReps

    def debug(self, flag=None):
        return self.base.debug(flag)

    def simulateOnce(self, RNG=None):
        datum = []
        for r in self.rReps:
            datum.append(self.base.simulateOnce(RNG))
        return datum

    def fillLikes(self):
        self.likes = []
        newlike = []
        for i, lk in enumerate(self.base.likes):
            if (i % self.rReps) or i==0:  # True if i divisible by rReps and i neq 0
                newlike.append(lk)
            else:
                self.likes.append(newlike)
                newlike = []

    def likelihoods(self, passedData=None):
        if passedData is None:
            self.fillLikes()
        return super(Repetitions, self).likelihoods(passedData)


    # def likelihoods(self, passedData=None):
    #     if passedData is None:
    #         concatenatedData = None  # use data stored in self.base
    #         mReps = self.base.mReps/self.rReps
    #     else:
    #         concatenatedData = []
    #         for datum in passedData:
    #             assert self.datumWellFormed(datum)
    #             concatenatedData += datum  # Expected that these are lists, see datumWellFormed
    #         mReps = len(concatenatedData)/self.rReps
    #     individualLikes = self.base.likelihoods(concatenatedData)  # if
    #     arrayLikes = numpy.array(individualLikes[0:mReps*self.rReps])
    #     arrayLikes = numpy.reshape(arrayLikes, (mReps, self.rReps))
    #     self.likes = arrayLikes.sum(axis=1).tolist()
    #     return self.likes

    def likeOnce(self, datum):
        assert self.datumWellFormed(datum)
        logLike = 0
        for datumComponent in datum:
            logLike += self.base.likeOnce(datumComponent)  # These are numbers
        return logLike

    def datumWellFormed(self, datum):
        mustBeTrue = len(datum) == self.rReps
        for d in datum:
            mustBeTrue = mustBeTrue and self.base.datumWellFormed(d)
        return mustBeTrue

    def mle(self):
        pass

    def lrN(self, alt, N, M):
        print "Don't call lrN from Reps class"
        assert False

    def aicN(self, alt, N, M):
        print "Don't call aicN from Reps class"
        assert False

