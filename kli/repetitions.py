__author__ = 'sean'
import numpy
import toy
import matplotlib.pylab as plt
import matplotlib
import scipy

class Repetitions(toy.FlatToy):
    def __init__(self, base, rReps, name=None):
        self.base = base
        self.rReps= rReps
        self.stack = []
        super(Repetitions, self).__init__(base, seed=None, name=name)

    def str_reps(self):
        return ' (%d reps of %s)' % (self.rReps, self.base.str_name())

    def __str__(self):
        first = super(Repetitions, self).__str__()
        last = self.str_reps()
        if len(first)+len(last) > 40:
            middle = '\n'
        else:
            middle = ' '
        return first + middle + last

    def defineRepetitions(self):
        pass

    def bootstrap(self, bReps, seed=None, RNG=None):
        self.bReps = bReps
        reps = bReps*self.rReps if bReps is not None else None
        self.set_base_mReps_to_mr(self.mReps, self.rReps)
        self.bootstrap_choice = self.base.bootstrap_choose(reps, seed, RNG)
        self.pop_base_mReps()

    def setUpExperiment(self, base, kw):
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
        baseLikesNoBootstrap = self.extendBaseLikes(trueModel)  # returns trueModel.base.likes
        if bootstrap:
            baseLikes = [baseLikesNoBootstrap[i] for i in trueModel.bootstrap_choice]
                         # for i in trueModel.bootstrap_choose(trueModel.bReps*trueModel.rReps)]
            # baseLikes = trueModel.resample(baseLikes, trueModel.bReps*trueModel.rReps)
            nLast = trueModel.bReps
        else:
            baseLikes = baseLikesNoBootstrap
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
                                   rRepsList=True, mReps=True, bReps=True, seed=None):
        if trueModel is None:
            trueModel = self
        if rRepsList is True:  # use trueModel (singleton list, default)
            rRepsList = [trueModel.rReps]  # Can pass longer list: e.g rRepsList=[1, 2, 3, 4, 5, 6]
        if mReps is True:  # use trueModel (default)
            mReps = trueModel.mReps
        if bReps is True:  # use trueModel (default)
            bReps = trueModel.bReps
        self.extendBaseLikes(trueModel, mReps, max(rRepsList))
        local_bootstrap_RNG = self.initRNG(seed)
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
        cv = self.base.likeRatioCV(alt.base, trueModel.base)
        trueModel.pop_base_mReps()
        return scipy.stats.norm.cdf(numpy.sqrt(self.rReps)/cv)

    def rInfinity(self, alt, trueModel=None, C=0.95):
        if trueModel is None:
            trueModel = self
        trueModel.set_base_mReps_to_mr()
        cv = self.base.likeRatioCV(alt.base, trueModel.base)
        trueModel.pop_base_mReps()



        return (scipy.stats.norm.ppf(C)*cv)**2

    def repeated_models(self, alt, trueModel=None, rReps=1, mReps=None, bReps=None):
        if trueModel is None:
            trueModel = self
        if mReps is True:
            mReps = trueModel.mReps
        if bReps is True:
            bReps = trueModel.bReps
        repeated_self = Repetitions(self.base, rReps)
        repeated_alt = Repetitions(alt.base, rReps)
        if trueModel is self:
            repeated_true = repeated_self
        elif trueModel is alt:
            repeated_true = repeated_alt
        else:
            repeated_true = Repetitions(trueModel.base, rReps)
        repeated_true.sim(mReps)  # if mReps is None then use all available data in trueModel.base
        repeated_true.bootstrap(bReps)  # if bReps is None, don't bootstrap
        return repeated_self, repeated_alt, repeated_true

    def desired_likelihood_ratio_coeff_variation(self, rMinus, pMinus):
        cv =  numpy.sqrt(rMinus)/scipy.stats.norm.ppf(pMinus)
        return cv

    def PrCurve(self, rMinus=None, pMinus=None, r=None, cv=None):
        if cv is None:
            cv = self.desired_likelihood_ratio_coeff_variation(rMinus, pMinus)
        PrPlus = scipy.stats.norm.cdf(numpy.sqrt(r)/cv) # = PrPlus
        return PrPlus

    def inversePrCurve(self, rMinus, pMinus, PrPlus):
        cv = self.desired_likelihood_ratio_coeff_variation(rMinus, pMinus)
        rPlus = (scipy.stats.norm.ppf(PrPlus)*cv)**2
        return rPlus

    def compute_initial_rMinus(self, alt, trueModel, C):
        rMinus = self.rInfinity(alt, trueModel, C)
        return max(1, int(rMinus))  # Make a positive integer

    def pos_integer(self, x):
        return max(1, int(x))

    def compute_pMinus(self, alt, trueModel=None, rMinus=None, C=0.95, mReps=True, bReps=True):
        if trueModel is None:
            trueModel = self
        if mReps is True:
            mReps = trueModel.mReps
        if bReps is True:
            bReps = trueModel.bReps
        assert mReps is not None
        if rMinus is None:
            rMinus = self.compute_initial_rMinus(alt, trueModel, C)
        repeated_self, repeated_alt, repeated_true = self.repeated_models(alt, trueModel,
                                                                          self.pos_integer(rMinus), mReps, bReps)
        pMinus = repeated_self.PFalsify(repeated_alt, repeated_true, adjustExtreme=True)
        return pMinus

    def rPlus(self, alt, trueModel=None, rMinus=None, pMinus=None, C=0.95, mReps=True, bReps=True):
        if rMinus is None:
            rMinus = self.compute_initial_rMinus(alt, trueModel, C)
        if pMinus is None:
            pMinus = self.compute_pMinus(alt, trueModel, rMinus, C, mReps, bReps)
        return self.inversePrCurve(rMinus, pMinus, C)

    def rMinus2Plus_plot(self, alt, trueModel, rMinus, pMinus, rPlus, C=0.95):
        if trueModel is None:
            trueModel = self
        r1 = max(2, int(rMinus))
        r2 = max(2, int(rPlus))
        rMin = min(r1-1, r2-1)
        rMax = max(r1+1, r2+1)
        rList = range(rMin, rMax)
        PFalsifyList = self.PFalsify_function_of_rReps(alt, trueModel, rRepsList=rList, mReps=True, bReps=True)
        rListFine = numpy.arange(rMin, rMax, .1)
        cv = self.desired_likelihood_ratio_coeff_variation(rMinus, pMinus)
        Probs = [self.PrCurve(r=r, cv=cv) for r in rListFine]
        plt.figure()
        ax = plt.gca()
        plt.plot(rList,PFalsifyList,'*', markersize=10)
        plt.plot(rListFine,Probs,'k:')
        (left, right, down, up) = plt.axis()
        plt.plot([rMinus,rMinus],[down,up],'r')
        plt.plot([rPlus,rPlus],[down,up],'g')
        reject = matplotlib.patches.Rectangle((left, down), right-left, C-down, color='red', alpha=.3)
        accept = matplotlib.patches.Rectangle((left, C), right-left, up-C, color='green', alpha=.3)
        ax.add_patch(reject)
        ax.add_patch(accept)
        plt.axis([left, right, down, up])

    def rStar(self, alt, trueModel=None, rMinus=None, C=0.95, mReps=True, bReps=True, iter=10, plot=False):
        for i in range(iter):
            rMinus = rPlus if i > 0 else rMinus
            pMinus = self.compute_pMinus(alt, trueModel, rMinus, C, mReps, bReps)
            rPlus = self.rPlus(alt, trueModel, rMinus, pMinus, C, mReps, bReps)
            # print "rMinus, pMinus, rPlus = ", rMinus, pMinus, rPlus
            print "Iteration: ", i, "| Value of R:", rMinus
        if plot:
            self.rMinus2Plus_plot(alt, trueModel, rMinus, pMinus, rPlus, C)
        return rMinus

    def lrN(self, alt, N, M):
        print "Don't call lrN from Reps class"
        assert False

    def aicN(self, alt, N, M):
        print "Don't call aicN from Reps class"
        assert False

