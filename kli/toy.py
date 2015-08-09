import numpy
import random
import time
import parameter
import matplotlib.pylab as plt
import matplotlib
import scipy.stats
import ad
import repository

class SaveStateRNG(numpy.random.RandomState):
    def __init__(self, seed=None):
        super(SaveStateRNG, self).__init__(seed)  # if seed=None, sets state by clock
        self.savedState = self.get_state()   # save initial state to use later

    def reseed(self, seed=None):
        self.seed(seed)  # if seed=None, sets state by clock
        self.savedState = self.get_state()   # save initial state to use later

    def reset(self):  # Resets RNG to same state as used first initialized or last time reseed() called.
        self.set_state(self.savedState)


class Toy(object):
    def __init__(self, q):
        self.q = q
        self.preferred = parameter.preferredUnits()
        self.preferred.time = 'milliseconds'

    def flatten(self, seed=None, name=None):
        parent = self  # for readability
        FT = FlatToy(parent, seed, name)
        return FT

    def getExperiment(self):  # For subclassing replace this code
        if len(self.q) == 1:
            toy2 = True
            q = parameter.mu(self.q[0], '1./'+self.preferred.time)
            q0 = None
            q1 = None
        elif len(self.q) == 2:
            toy2 = False
            q = None
            q0 = parameter.mu(self.q[0], '1/'+self.preferred.time)
            q1 = parameter.mu(self.q[1], '1/'+self.preferred.time)
        else:
            assert (False)  # Length of q should be 1 or 2
        return (toy2, q, q0, q1)


class FlatToy(object):
    def __init__(self, parent, seed=None, name=None):
        self.debugFlag = False  # To save hidden states, call self.debug() before generating data
        self.R = self.initRNG(seed)
        self.setUpExperiment(parent)
        self.defineRepetitions()
        self.startData()
        self.bootstrap(None)
        self.startLikes()
        self.rename(name)

    def rename(self, name=None):
        if name is None:
            name = repr(self)
        self.name = name

    def str_name(self):  # for overloading in repetitions
        if self.name is None:
            return repr(self)
        else:
            return self.name

    def str_hat(self, alt, trueModel):
        if trueModel is None or trueModel is self:
            trueModel = self
            tru_name = " = H"
        elif trueModel is alt:
            trueModel = alt
            tru_name = " = A"
        else:
            tru_name = ": " + trueModel.str_name() + trueModel.str_reps()
        hyp_name = self.str_name() + self.str_reps()
        alt_name = alt.str_name() + alt.str_reps()
        return 'H: '+hyp_name+' A: '+alt_name+'\nT'+tru_name+trueModel.str_mb()

    def str_mb(self):
        return ": m=%s, b=%s" % (str(self.mReps), str(self.bReps))

    def str_reps(self):
        return ""

    def __str__(self):
        return self.str_name() + self.str_mb()

    def defineRepetitions(self):
        self.base = self  # Used in functions below; Defined differently for Repetitions subclass
        self.rReps = 1  # Used in functions below; Defined differently by Repetitions subclass

    def bootstrap_choose(self, bReps, seed=None, RNG=None):
        # Pass either seed (for new RNG) or RNG
        if bReps is None:
            self.bootstrap_choice = []
            return None
        if RNG is None:
            RNG = self.initRNG(seed)
        return RNG.choice(range(self.mReps), bReps).tolist()

    def bootstrap(self, bReps, seed=None, RNG=None):
        self.bReps = bReps
        self.bootstrap_choice = self.bootstrap_choose(bReps, seed, RNG)

    def initRNG(self, seed=None):  # Maybe overloaded if using a different RNG, eg rpy2
        return SaveStateRNG(seed)

    def startData(self):
        self.data = []  # Data used for fitting model. (Each datum may be a tuple)
        self.hiddenStates = []  # These are the Markov states, including hidden ones.  This model isn't Markovian, though
        self.mReps = 0

    def startLikes(self):
        self.likes = repository.TableOfModels()
        self.likeInfo = repository.TableOfModels()

    def _restart(self):  # Clears data and resets RNG with same seed
        self.R.reset()
        self.startData()
        self.startLikes()
        print 'WARNING: Method only intended for debugging.  Not Fully supported.'

    def _reseed(self, seed=None):
        self.R.reseed(seed)
        self._restart()

    def setUpExperiment(self, parent):
        self.experiment = parent.getExperiment()
        self.toy2, self.q, self.q0, self.q1 = self.experiment

    def _changeModel(self, parent):
        self.setUpExperiment(parent)
        self._restart()

    def sim(self, mReps=None):  # Only does new reps; keeps old; if (nReps < # Trajs) then does nothing
        if mReps is None:
            mReps = len(self.data)
        numNewReps = mReps - len(self.data)  # Negative if decreasing nReps; if so, nReps updated data unchanged
        for n in range(numNewReps):
            self.data.append(self.simulateOnce(self.R))  # Don't want to use self.R elsewhere
            if self.debugFlag:
                self.hiddenStates.append(self.hiddenStateTrajectory)
        self.mReps = mReps  # Might be decreasing nReps, but code still saves the old results

    def resim(self, mReps=0):
        self.R.reset()
        self.startData()
        self.sim(mReps)

    def trim(self, mReps=0):
        self.likes.trim(mReps)
        self.likeInfo.trim(mReps)

    def debug(self):
        assert(len(self.data) == 0)   # Can't have generated any data; use "self._debug(True)" to override
        self.debugFlag = True

    def _debug(self, flag=None):
        if flag == True:
            self.debugFlag = True
            self._restart()  # Restart because you need to rerun to save hidden states
        elif flag == False:
            self.debugFlag = False
        return self.debugFlag

    def simulateOnce(self, RNG=None):  # Overload
        if RNG is None:
            RNG = self.initRNG(None)
        if self.toy2:
            self.hiddenStateTrajectory = (RNG.exponential(1./self.q),)  # Though not Markovian, we can save the hidden transition times
        else:
            self.hiddenStateTrajectory = (RNG.exponential(1./self.q1), RNG.exponential(1./self.q0))
        return sum(self.hiddenStateTrajectory)

    def likelihoods_monte_carlo_sample(self, trueModel=None):
        if trueModel is None:  # Data not passed
            trueModel = self
        likes = trueModel.likes.getOrMakeEntry(self)
        nFirst = len(likes)
        nLast = trueModel.mReps  # Restricts return to self.mReps
        if nLast == 0:  # No data in true model
            print "Warning: Must Have Data in True Model; No Likelihoods"
            return None
        for datum in trueModel.data[nFirst:nLast]:
            likes.append(self.likeOnce(datum))
            # if self.debugFlag and self.recentLikeInfo is not None:
            #    likeInfo.append(self.recentLikeInfo)
        return likes[0:nLast]  # Restrict what you return to stopping point

    def likelihoods_bootstrap_sample(self, trueModel=None):
        likes = self.likelihoods_monte_carlo_sample(trueModel)
        return [likes[i] for i in trueModel.bootstrap_choice]

    def likelihoods(self, trueModel=None):
        if trueModel is None:
            trueModel = self
        if trueModel.bReps is None:
            return self.likelihoods_monte_carlo_sample(trueModel)
        else:
            return self.likelihoods_bootstrap_sample(trueModel)

    def likeOnce(self, datum):  # Overload when subclassing
        #if self.debugFlag:
        #    self.recentLikeInfo = None
        if not self.datumIntegrity(datum):
            return -numpy.infty
        elif self.toy2:
            return numpy.log(self.q) - self.q * datum
        elif self.q0 == self.q1:
            return numpy.log(self.q1) + numpy.log(self.q0) - self.q0 * datum + numpy.log(datum)
        else:
            return numpy.log(self.q1) + numpy.log(self.q0) + numpy.log(
                (numpy.exp(-self.q0 * datum) - numpy.exp(-self.q1 * datum)) / (self.q1 - self.q0))

    def datumWellFormed(self, datum):
        return isinstance(datum, float) or isinstance(datum, int)

    def datumIntegrity(self, datum):
        if not self.datumWellFormed(datum):
            return False
        elif datum < 0.:
            return False
        elif (not self.toy2) and datum == 0:
            return False
        else:
            return True

    def minuslike(self, trueModel=None):
        L = self.likelihoods(trueModel)
        return -sum(L)

    def like(self, trueModel=None):
        L = self.likelihoods(trueModel)
        return sum(L)

    def pdf(self, datum):
        return numpy.exp(self.likeOnce(datum))

    # def mle(self):
    #     assert self.toy2  # Not yet implemented for toy 3
    #     return 1. / numpy.mean(self.data[0:self.mReps])

    def logf(self, trueModel=None):
        return numpy.matrix(self.likelihoods(trueModel))

    def likeRatios(self, alt, trueModel=None):  # likelihood ratio; self is true model
        if trueModel is None:
            trueModel = self  # if true=None, want alt(hyp) not alt(alt), below
        return self.logf(trueModel) - alt.logf(trueModel)

    def likeRatioHistogram(self, alt, trueModel=None, bins=10):
        likelihood_ratios = self.likeRatios(alt, trueModel)
        plt.figure()
        ax = plt.gca()
        plt.hist(likelihood_ratios.T, bins=bins, normed=True, color='black')
        (left, right, down, up) = plt.axis()
        reject = matplotlib.patches.Rectangle((left, down), 0-left, up-down, color='red', alpha=.3)
        accept = matplotlib.patches.Rectangle((0, down), right-0, up-down, color='green', alpha=.3)
        ax.add_patch(reject)
        ax.add_patch(accept)
        plt.hist(likelihood_ratios.T, bins=bins, normed=True, color='black')
        plt.axis([left, right, down, up])
        plt.xlabel("Likelihood Ratio")
        plt.ylabel('Density of Likelihood Ratios')
        plt.title(self.str_hat(alt, trueModel))

    def PFalsify(self, alt, trueModel=None, adjustExtreme=False):
        ratios = self.likeRatios(alt, trueModel)
        number_of_ratios = ratios.shape[1]
        if number_of_ratios == 0:
            print "Warning: No Likelihoods"
            return None
        if self.debugFlag:
            print "number of ratios =", number_of_ratios
            print "ratios =", ratios
        number_of_positives = numpy.sum(ratios > 0)
        if adjustExtreme and number_of_positives == 0:
            number_of_positives += 0.5
        elif adjustExtreme and number_of_positives==number_of_ratios:
            number_of_positives -= 0.5
        return number_of_positives/float(number_of_ratios)

    def likeRatioMuSigma(self, alt, trueModel=None):  # self is true model
        lrs = self.likeRatios(alt, trueModel)
        mu = numpy.mean(lrs)
        sig = numpy.std(lrs)
        return mu, sig

    def likeRatioCV(self, alt, trueModel=None):
        mu, sig = self.likeRatioMuSigma(alt, trueModel)
        return sig/mu

    # Deprecated and will be removed in future.  Need to update sfn14.
    def lrN(self, alt, N, M, trueModel=None):  # add N of them, return M
        self.sim(mReps=N * M)
        lrNM = self.likeRatios(alt, trueModel)
        L = numpy.reshape(lrNM, (M, N))
        return L.sum(axis=0)

    def aic(self, alt, trueModel=None):  # self is true model
        if trueModel is None:
            trueModel = self
        return 2 * (self.logf(trueModel) - alt.logf(trueModel))

    def aicMuSigma(self, alt, trueModel=None):  # self is true model
        aics = self.aic(alt, trueModel)
        mu = numpy.mean(aics)
        sigma = numpy.std(aics)
        return (mu, sigma)

    # Deprecated and will be removed in future.  Need to update sfn14,
    def aicN(self, alt, N, M, trueModel=None):  # add N of them, return M
        self.sim(mReps=N * M)
        aicNM = self.aic(alt)
        A = numpy.reshape(aicNM, (M, N))
        return A.sum(axis=0)

    def Ehlogf(self, trueModel=None):
        return (self.logf(trueModel).mean())

    def KL(self, other, trueModel=None):
        if trueModel is None:
            trueModel = self
        return self.Ehlogf(trueModel) - other.Ehlogf(trueModel)


class likefun(object):
    def __init__(self, parent, paramTuple):
        self.parent = parent
        self.paramTuple = paramTuple
        self.F = self.parent.flatten()

    def set(self, valueTuple):
        for i, P in enumerate(self.paramTuple):
            P.assign(valueTuple[i])
        # Ex = self.parent.getExperiment()
        self.F._changeModel(self.parent)
        # self.F.changeModel(self.parent)

    def setLog(self, valueTuple):
        for i, P in enumerate(self.paramTuple):
            P.assignLog(valueTuple[i])  # AssignLog so that assigned values can vary from -infty to infty
        # Ex = self.parent.getExperiment()
        self.F._changeModel(self.parent)
        # self.F.changeModel(self.parent)

    def sim(self, XTrue, nReps=100, seed=None, log=True):
        self.XTrue = XTrue
        if log == True:
            self.setLog(XTrue)
        else:
            self.set(XTrue)
        self.F._reseed(seed)
        self.F.sim(nReps, clear=True)  # clear=True should now be redundant, but kept here for readability

    def minuslike(self, x):
        self.setLog(x)
        # if x[0] < 0. or x[1]<0:
        #    print "x is negative"
        #print "x[0], q[0]", x[0], q0.value
        #print "x[1], q[1]", x[1], q1.value
        return self.F.minuslike()

    def like(self, x, log=True):
        if log == True:
            self.setLog(x)
        else:
            self.set(x)
        return self.F.like()

q0 = parameter.Parameter("q0", 0.5, "kHz", log=True)
q1 = parameter.Parameter("q1", 0.25, "kHz", log=True)
q = parameter.Parameter("q", 1. / 6., "kHz", log=True)
T3 = Toy([q0, q1])
T2 = Toy([q])
FT3 = T3.flatten(seed=3)
FT2 = T2.flatten(seed=3)
XRange = numpy.arange(0.1, 30.1, 1)
YRange = numpy.arange(0.11, 30.11, 1)  # Different values so rate constants remain unequal
