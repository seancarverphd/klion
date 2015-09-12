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

    def datumIntegrity(self, datum):
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
        # prob = 0
        # for i, t in enumerate(testStatistic):
        #     if t > 0:
        #         prob += self.B.pmf(i)
        # return prob

if __name__ == '__main__':
    S20 = Simple(n=20, p=.5)
    ES20 = S20.exact()
    FS20 = S20.flatten(name='FS20')
    R20 = repetitions.Repetitions(FS20,9,name='R20')
    B9 = Simple(n=1, p=.9)
    B8 = Simple(n=1, p=.8)
    EB9 = B9.exact()
    FB9 = B9.flatten(name='Bern.9')
    EB8 = B8.exact()
    FB8 = B8.flatten(name='Bern.8')
    S21 = Simple(n=21, p=.5)
    S19 = Simple(n=19, p=.5)
    ES21 = S21.exact()
    FS21 = S21.flatten(name='FS21')
    R21 = repetitions.Repetitions(FS21,9,name='R21')
