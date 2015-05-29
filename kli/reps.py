__author__ = 'sean'
import numpy
import toy


class Reps(toy.FlatToy):
    def __init__(self, base, reps):
        self.reps = reps
        super(Reps, self).__init__(base)

    def initRNG(self, seed=None):
        pass

    def setUpExperiment(self, base):
        self.base = base
        # No changedSinceLastSim, don't want to clear base

    def reseed(self, seed=None):
        self.base.reseed(seed)

    def restart(self):
        self.base.restart()

    def sim(self, nReps=1, clear=False):
        if clear:
            self.restart()
        # No changedSinceLastSim, don't want to clear base unless explicitly directed
        self.base.sim(self.reps*nReps)  # only computes newly needed data
        self.nReps = nReps

    def debug(self, flag=None):
        return self.base.debug(flag)

    def simulateOnce(self, RNG=None):
        datum = []
        for r in self.reps:
            datum.append(self.base.simulateOnce(RNG))
        return datum

    def likelihoods(self, passedData=None):
        if passedData is None:
            concatenatedData = None  # use data stored in self.base
        else:
            concatenatedData = []
            for datum in passedData:
                assert self.datumWellFormed(datum)
                concatenatedData += datum  # Expected that these are lists, see datumWellFormed
        individualLikes = self.base.likelihoods(concatenatedData)  # if
        likereps = len(concatenatedData)/self.reps
        arrayLikes = numpy.array(individualLikes[0:likereps*self.reps])
        arrayLikes = numpy.reshape(arrayLikes, (likereps, self.reps))
        return arrayLikes.sum(axis=0).tolist()

    def likeOnce(self, datum):
        assert datumWellFormed(datum)
        logLike = 0
        for datumComponent in datum:
            logLike += self.base.likeOnce(datumComponent)  # These are numbers
        return logLike

    def datumWellFormed(self, datum):
        mustBeTrue = len(datum) == self.reps
        for d in datum:
            mustBeTrue = mustBeTrue and self.base.datumWellFormed(d)
        return mustBeTrue

    def mle(self):
        pass

    def nRepsRestrictedData(self):
        assert self.base.nReps >= self.reps * self.nReps
        data = []
        for r in range(self.nReps):
            datum = []
            for d in range(self.reps):
                datum.append(d)
            data.append(datum)
        return data
    

    def lrN(self, alt, N, M):
        print "Don't call lrN from Reps class"
        assert False

    def aicN(self, alt, N, M)
        print "Don't call aicN from Reps class"
        assert False

