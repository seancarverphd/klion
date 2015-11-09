__author__ = 'sean'
import scipy.stats
import toy
import parameter
import repetitions


class Simple(object):
    def __init__(self, n=1, p=.5):
        self.n = n
        self.p = p
        self.preferred = parameter.preferredUnits()

    def flatten(self, seed=None, name=None):
        parent = self  # for readability
        return FlatSimple(parent, seed, name)

    def exact(self):
        parent = self  # for readability
        return ExactSimple(parent)

    def getExperiment(self):
        n = int(parameter.mu(self.n, 'dimensionless'))
        p = parameter.mu(self.p, 'dimensionless')
        return {'n': n, 'p': p}


class FlatSimple(toy.FlatToy):
    def unpackExperiment(self):
        self.n = self.experiment['n']
        self.p = self.experiment['p']
        self.B = scipy.stats.binom(self.n, self.p)

    def simulateOnce(self, RNG=None):
        if RNG is None:
            RNG = self.initRNG(None)
        return RNG.binomial(self.n, self.p)

    def likeOnce(self, datum):
        return self.B.logpmf(datum)

    def datumWellFormed(self,datum):
        return isinstance(datum, int)

    def datumSupported(self, datum):
        return self.datumWellFormed(datum) and (datum <= self.n) and (datum >= 0)


class ExactSimple(object):
    def __init__(self, parent):
        self.experiment = parent.getExperiment()
        self.n = self.experiment['n']
        self.p = self.experiment['p']
        self.B = scipy.stats.binom(self.n, self.p)

    def Elogf(self, trueModel=None):
        if trueModel is None or trueModel is self:
            return -self.B.entropy()
        else:
            n_true = trueModel.B.args[0]
            return sum([self.B.logpmf(i)*trueModel.B.pmf(i)
                        for i in range(n_true + 1)])  # range of binomial extends to n_true not just n_true - 1

    def KL(self, other, true_model=None):
        if true_model is None:
            true_model = self
        return self.Elogf(true_model) - other.Elogf(true_model)

    def PFalsify(self, other, true_model=None):
        if true_model is None:
            true_model = self
        n_true = true_model.B.args[0]
        test_statistic = [self.B.logpmf(i) - other.B.logpmf(i)
                          for i in range(n_true + 1)]  # range of binomial extends to n_true not just n_true - 1
        probabilities_of_correct_selection = [true_model.B.pmf(i) if test_statistic[i] > 0 else 0
                                              for i in range(n_true + 1)]  # range of binomial extends to n_true
        return sum(probabilities_of_correct_selection)

class ExactSimpleRepetitions(object):
    def __init__(self, parent, r):
        self.parent = parent
        self.n = parent.n
        self.p = parent.p
        self.r = r
        self.index, self.x, self.pmf = self.construct()

    def construct(self):
        # data are n-vectors where ith element is number of times out of r i channels seen open
        x0 = [self.r]+[0]*self.n
        x_enumerated = self.multi_enumerate([])
        index = {}
        for i, x in enumerate(x_enumerated):
            index[x] = i
        logpmf_enumerated = self.get_logpmf(x_enumerated)
        return index, x_enumerated, logpmf_enumerated

    def multi_enumerate(self, prefix):
        x_enumerated = []
        if len(prefix) == self.n:
            return [tuple(prefix + [self.r-sum(prefix)])]
        for k in range(self.r-sum(prefix)+1):
            x_enumerated += self.multi_enumerate(prefix+[k])
        return x_enumerated

    def get_logpmf(self, x_enumerated):
        for x in x_enumerated:
            for i, xi in enumerate(x):sym
if __name__ == '__main__':
    RootS20 = Simple(n=20, p=.5)
    ES20 = RootS20.exact()
    S20 = RootS20.flatten(name='FS20')
    R20 = repetitions.Repetitions(S20,9,name='R20')
    RootB9 = Simple(n=1, p=.9)
    RootB8 = Simple(n=1, p=.8)
    EB9 = RootB9.exact()
    B9 = RootB9.flatten(name='Bern.9')
    EB8 = RootB8.exact()
    B8 = RootB8.flatten(name='Bern.8')
    RootS21 = Simple(n=21, p=.5)
    RootS19 = Simple(n=19, p=.5)
    ES21 = RootS21.exact()
    S21 = RootS21.flatten(name='FS21')
    R21 = repetitions.Repetitions(S21,9,name='R21')
