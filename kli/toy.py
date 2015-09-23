import numpy
import copy
import matplotlib.pylab as plt
import matplotlib
import repository
import parameter

class SaveStateRNG(numpy.random.RandomState):
    def __init__(self, seed=None):
        super(SaveStateRNG, self).__init__(seed)  # if seed=None, sets state by clock
        self.savedState = self.get_state()   # save initial state to use later

    def reseed(self, seed=None):
        if seed is False:
            return
        self.seed(seed)  # if seed=None, sets state by /dev/urandom
        self.savedState = self.get_state()   # save initial state to use later

    def restate(self, seed_or_state):
        try:
            self.set_state(seed_or_state)
        except TypeError:
            self.seed(seed_or_state)
        self.savedState = self.get_state()

    def reset(self):  # Resets RNG to same state as used first initialized or last time reseed() called.
        self.set_state(self.savedState)


class Select(object):
    def __init__(self, parent, bReps=True, mReps=True, seed_or_state=None):
        self.bReps, self.mReps = self.process_reps(parent, bReps, mReps)
        if bReps is None:
            self.seed_or_state = None
            self.RNG = None
            self.choice = range(mReps)
        else:
            self.seed_or_state = seed_or_state
            self.RNG = SaveStateRNG()
            self.RNG.restate(seed_or_state)
            self.choice = self.RNG.choice(range(mReps), bReps).tolist()

    def process_reps(self, parent, bReps, mReps):
        if bReps is True:
            bReps = parent.bReps
        if mReps is True:
            mReps = parent.mReps
        if mReps is None:
            mReps = parent.mTotal()
        return bReps, mReps

    # def unchanged(self, bReps, mReps, seed_or_state):
    #     return (bReps == self.bReps and mReps == self.mReps and seed_or_state is not None and
    #             seed_or_state == self.seed_or_state)

class Toy(object):
    def __init__(self, qs):
        self.qs = qs
        self.preferred = parameter.preferredUnits()
        self.preferred.time = 'milliseconds'

    def flatten(self, seed=None, name=None):
        parent = self  # for readability
        FT = FlatToy(parent, seed, name)
        return FT

    def getExperiment(self):  # For subclassing replace this code
        assert len(self.qs) == 1 or len(self.qs) == 2  # have not implemented other toy models
        if len(self.qs) == 1:
            toy2 = True
            q = parameter.mu(self.qs[0], '1./'+self.preferred.time)
            q0 = None
            q1 = None
        else:  # len(self.qs) == 2:
            toy2 = False
            q = None
            q0 = parameter.mu(self.qs[0], '1/'+self.preferred.time)
            q1 = parameter.mu(self.qs[1], '1/'+self.preferred.time)
        return {'toy2':toy2, 'q':q, 'q0':q0, 'q1':q1}


class FlatToy(object):
    def __init__(self, parent, seed=None, name=None, kw=None):
        self.debugFlag = False  # To save hidden states, call self.debug() before generating data
        self.simRNG = self.initRNG(seed)
        self.selection = None
        self.setUpExperiment(parent, kw)
        self.defineRepetitions()
        self.startData()
        self.startLikes()
        self.rename(name)

    def getExperiment(self):
        return self.experiment

    def setUpExperiment(self, parent, kw):
        self.experiment = parent.getExperiment()
        if kw is not None:
            original_experiment_length = len(self.experiment)
            self.experiment.update(kw)
            assert original_experiment_length == len(self.experiment)  # can't add fields that aren't already there
        self.unpackExperiment()

    def unpackExperiment(self):
        self.toy2 = self.experiment['toy2']
        self.q = self.experiment['q']
        self.q0 = self.experiment['q0']
        self.q1 = self.experiment['q1']

    def spawn(self, seed=None, name=None, **kw):
        parent = self  # for readability
        return type(self)(parent, seed, name, kw)

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

    # def same_bootstrap(self, bReps, mReps, seed):
    #     if seed is None:
    #         return False
    #     else:
    #         return (mReps == self.bootstrap_previous_mReps
    #             and bReps == self.bootstrap_previous_bReps
    #             and seed == self.bootstrap_previous_seed)

    def mTotal(self):  # overloaded for Repetitions class
        return len(self.data)

    # def select_from_reps(self, bReps, mReps, selector_seed_or_state=True):
    #     if bReps is True:
    #         bReps = self.bReps
    #     if mReps is True:
    #         mReps = self.mReps
    #     elif mReps is None:
    #         mReps = self.mTotal()
    #     return self.selection if self.selection.unchanged(bReps, mReps, selector_seed_or_state) \
    #                           else Selector(bReps, mReps, selector_seed_or_state)

    # def bootstrap_choose(self, bReps=True, mReps=True, seed=None, RNG=None):
    #     bReps, mReps = self.process_default_reps(bReps, mReps)
    #     if RNG is None:
    #         RNG = self.initRNG()
    #     RNG.reseed(seed)  # if seed is False, does nothing.
    #     self.extend_data(mReps=mReps)
    #     if bReps is None:
    #         return []
    #     else:
    #         return RNG.choice(range(mReps), bReps).tolist()

    def bootstrap(self, bReps=True, mReps=True, selector_seed_or_state=None):
        self.selection = Select(self, bReps, mReps, selector_seed_or_state)
        self.extend_data(self.selection.mReps)
        self.bReps = self.selection.bReps
        self.mReps = self.selection.mReps

    def initRNG(self, seed=None):  # Maybe overloaded if using a different RNG, eg rpy2
        return SaveStateRNG(seed)

    def startData(self):
        self.data = []  # Data used for fitting model. (Each datum may be a tuple)
        self.hiddenStates = []  # These are the Markov states, including hidden ones.  This model isn't Markovian, though
        self.bReps = None
        self.mReps = 0
        # self.bootstrap_previous_bReps = None
        # self.bootstrap_previous_mReps = None
        # self.bootstrap_previous_seed = None
        self.bootstrap_choice = []

    def startLikes(self):
        self.likes = repository.TableOfModels()
        self.likeInfo = repository.TableOfModels()

    def _restart(self):  # Clears data and resets RNG with same seed
        self.simRNG.reset()
        self.startData()
        self.startLikes()
        print 'WARNING: Method only intended for debugging.  Not Fully supported.'

    def _reseed(self, seed=None):
        self.simRNG.reseed(seed)
        self._restart()

    def _changeModel(self, parent, kw):
        self.setUpExperiment(parent, kw)
        self._restart()

    def extend_data(self, mReps=True):  # New reps added, keeps old; if (mReps <= len(self.data) then does nothing
        if mReps is True or mReps is None:
            return
        numNewReps = mReps - len(self.data)  # Nothing changed if negative
        for n in range(numNewReps):
            self.data.append(self.simulateOnce(self.simRNG))  # Don't want to use self.R elsewhere
            if self.debugFlag:
                self.hiddenStates.append(self.hiddenStateTrajectory)

    def sim(self, mReps=None):
        self.bootstrap(None, mReps)
        # if mReps is True:
        #     return
        # self.extend_data(mReps)
        # self.mReps = self.mTotal() if mReps is None else mReps

    def resim(self, mReps=0):
        self.simRNG.reset()
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

    def likelihoods(self, trueModel=None, selection=True):
        if trueModel is None:  # Data not passed
            trueModel = self
        if selection is True:
            selection = trueModel.selection
        elif selection is None:
            print "Must have data in true model"
            return []
        likes = self.extend_likes(trueModel, selection.mReps)
        return [likes[i] for i in selection.choice]

    def process_mReps(self, mReps=True):
        if mReps is True:
            return self.mReps
        elif mReps is None:
            return self.mTotal()
        else:
            return mReps

    # def process_trueModel(self, trueModel=None):
    #     if trueModel is None:
    #         return self
    #     else:
    #         return trueModel

    def extend_likes(self, trueModel=None, mReps=True):
        if trueModel is None:
            trueModel = self
        mReps = trueModel.process_mReps(mReps)
        trueModel.extend_data(mReps)
        likes = trueModel.likes.getOrMakeEntry(self)
        mFirst = len(likes)
        mLast = mReps
        assert(mReps > 0)  # Must have some data to run likelihoods
        for datum in trueModel.data[mFirst:mLast]:
            likes.append(self.likeOnce(datum))
        return likes

    def get_data(self, mReps=True):
        mReps = self.process_mReps(mReps)
        self.extend_data(mReps)
        return self.data[0:mReps]

    def get_likes(self, trueModel=None, mReps=True):
        if trueModel is None:
            trueModel = self
        mReps = trueModel.process_mReps(mReps)
        return self.extend_likes(trueModel, mReps)

    def get_bootstrap_data(self, selection=None):
        if selection is None:
            selection = self.selection
        return [self.data[i] for i in selection.choice]

    def get_bootstrap_likes(self, trueModel=None, selection=None):
        if trueModel is None:
            trueModel = self
        if selection is None:
            selection = trueModel.selection
        likes = self.get_likes(trueModel, selection.mReps)
        return [likes[i] for i in selection.choice]

    # def likelihoods_bootstrap_sample(self, trueModel=None, bReps=True, mReps=True):
    #     likes = self.likelihoods_monte_carlo_sample(trueModel, mReps=trueModel.mReps)
    #     # return [likes[i] for i in trueModel.bootstrap_choose(bReps, mReps)]
    #     return [likes[i] for i in trueModel.bootstrap_choice]

    # def likelihoods(self, trueModel=None, bReps=True, mReps=True):
    #     if trueModel is None:
    #         trueModel = self
    #     bReps, mReps = trueModel.process_default_reps(bReps, mReps)
    #     if bReps is None:
    #         return self.likelihoods_monte_carlo_sample(trueModel, mReps=mReps)
    #     else:
    #         return self.likelihoods_bootstrap_sample(trueModel, bReps=bReps, mReps=mReps)

    def likeOnce(self, datum):  # Overload when subclassing
        if not self.datumSupported(datum):
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

    def datumSupported(self, datum):
        if not self.datumWellFormed(datum):
            return False
        elif datum < 0.:
            return False
        elif (not self.toy2) and datum == 0:
            return False
        else:
            return True

    def minuslike(self, trueModel=None, selection=True):
        L = self.likelihoods(trueModel, selection)
        return -sum(L)

    def like(self, trueModel=None, selection=True):
        L = self.likelihoods(trueModel, selection)
        return sum(L)

    def pdf(self, datum):
        return numpy.exp(self.likeOnce(datum))

    def pdf_of_iterable(self, x_iterable):
        density_curve = copy.copy(x_iterable)
        for i, datum in enumerate(x_iterable):
            density_curve[i] = self.pdf(datum)
        return density_curve

    def compare_pdfs(self, alt, dataModel=None, a=None, b=None, n=1000, bins=10, xlab=None):
        assert not (dataModel is None and (a is None or b is None))
        if dataModel is True:
            dataModel = self
        plt.figure()
        plt.hold('off')
        if dataModel is not None:
            plt.hist(numpy.array(dataModel.data[0:self.mReps]), bins=bins, normed=True, color='black', alpha=0.4)
            plt.hold('on')
            ax_left, ax_right, ax_lo, ax_hi = plt.axis()
            if a is None:
                a = ax_left
            if b is None:
                b = ax_right
        x_iterable = numpy.linspace(a, b, n)
        density_curve_true = self.pdf_of_iterable(x_iterable)
        density_curve_alt = alt.pdf_of_iterable(x_iterable)
        index_boundaries = numpy.where(numpy.diff(density_curve_true > density_curve_alt))[0] + .5
        select_true = density_curve_true[0] > density_curve_alt[0]
        x_boundaries = a + index_boundaries*(b-a)/(n-1)
        left_endpoints = numpy.concatenate((numpy.array([a]), x_boundaries))
        right_endpoints = numpy.concatenate((x_boundaries, numpy.array([b])))
        pdf_true = self.pdf_of_iterable(x_iterable)
        pdf_alt = alt.pdf_of_iterable(x_iterable)
        plt.plot(x_iterable, pdf_true, 'b-')
        plt.hold('on')
        plt.plot(x_iterable, pdf_alt, 'b--')
        axes_left, axes_right, axes_lo, axes_hi = plt.axis()
        for i in range(len(left_endpoints)):
            if select_true:
                col = 'green'
            else:
                col = 'red'
            region = matplotlib.patches.Rectangle((left_endpoints[i],axes_lo),
                                                  right_endpoints[i]-left_endpoints[i],
                                                  axes_hi-axes_lo, color=col, alpha=.3)
            ax = plt.gca()
            ax.add_patch(region)
            select_true = not select_true
        # plt.xlabel("Channel Opening Time (ms)")
        plt.ylabel("Probability Density")
        plt.legend((self.name, alt.name), loc=1)
        if xlab is not None:
            plt.xlabel(xlab)
        plt.show()

    def pdf_plot(self, x_iterable):
        density_curve = self.pdf_of_iterable(x_iterable)
        plt.figure()
        # ax = plt.gca()
        plt.plot(x_iterable, density_curve)

    # def mle(self):
    #     assert self.toy2  # Not yet implemented for toy 3
    #     return 1. / numpy.mean(self.data[0:self.mReps])

    def logf(self, trueModel=None, selection=True):
        return numpy.matrix(self.likelihoods(trueModel, selection))

    def likeRatios(self, alt, trueModel=None, selection=True):  # likelihood ratio; self is true model
        if trueModel is None:
            trueModel = self  # if true=None, want alt(hyp) not alt(alt), below
        if selection is True:
            selection = trueModel.selection
        return self.logf(trueModel, selection) - alt.logf(trueModel, selection)

    def dataHistogram(self, bins=10):
        plt.figure()
        plt.gca()
        plt.hist(numpy.array(self.data[0:self.mReps]), bins=bins, normed=True, color='black')

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

    def PFalsify(self, alt, trueModel=None, selection=True, adjustExtreme=False):
        ratios = self.likeRatios(alt, trueModel, selection)
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

    def likeRatioMuSigma(self, alt, trueModel=None, selection=True):  # self is true model
        lrs = self.likeRatios(alt, trueModel, selection)
        mu = numpy.mean(lrs)
        sig = numpy.std(lrs)
        return mu, sig

    def likeRatioCV(self, alt, trueModel=None, selection=True):
        mu, sig = self.likeRatioMuSigma(alt, trueModel, selection)
        return sig/mu

    # Deprecated and will be removed in future.  Need to update sfn14.
    def lrN(self, alt, N, M, trueModel=None, selection=True):  # add N of them, return M
        self.sim(mReps=N * M)
        lrNM = self.likeRatios(alt, trueModel, selection)
        L = numpy.reshape(lrNM, (M, N))
        return L.sum(axis=0)

    def aic(self, alt, trueModel=None, selection=True):  # self is true model
        if trueModel is None:
            trueModel = self
        if selection is True:
            selection = trueModel.selection
        return 2 * (self.logf(trueModel, selection) - alt.logf(trueModel, selection))

    def aicMuSigma(self, alt, trueModel=None, selection=True):  # self is true model
        aics = self.aic(alt, trueModel, selection)
        mu = numpy.mean(aics)
        sigma = numpy.std(aics)
        return (mu, sigma)

    # Deprecated and will be removed in future.  Need to update sfn14,
    def aicN(self, alt, N, M, trueModel=None):  # add N of them, return M
        self.sim(mReps=N * M)
        aicNM = self.aic(alt)
        A = numpy.reshape(aicNM, (M, N))
        return A.sum(axis=0)

    def Ehlogf(self, trueModel=None, selection=True):
        return (self.logf(trueModel, selection).mean())

    def KL(self, other, trueModel=None, selection=True):
        if trueModel is None:
            trueModel = self
        if selection is True:
            selection = trueModel.selection
        # ORIGINALLY (less stable?):  return self.Ehlogf(trueModel) - other.Ehlogf(trueModel)
        return (self.logf(trueModel, selection) - other.logf(trueModel, selection)).mean()


if __name__ == '__main__':
    q0 = parameter.Parameter("q0", 0.5, "kHz", log=True)
    q1 = parameter.Parameter("q1", 0.25, "kHz", log=True)
    q = parameter.Parameter("q", 1. / 6., "kHz", log=True)
    T3 = Toy([q0, q1])
    T2 = Toy([q])
    FT3 = T3.flatten(seed=3)
    FT2 = T2.flatten(seed=3)
    XRange = numpy.arange(0.1, 30.1, 1)
    YRange = numpy.arange(0.11, 30.11, 1)  # Different values so rate constants remain unequal
