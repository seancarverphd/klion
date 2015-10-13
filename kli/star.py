__author__ = 'sean'

import numpy as np
import toy

class Star(object):
    def __init__(self, hyp, alt, trueModel=None, mReps=None, seed=None):
        self.RNG = toy.SaveStateRNG(seed)
        self.hyp = hyp
        self.alt = alt
        if trueModel is None:
            self.trueModel = hyp
        else:
            self.trueModel = trueModel
        assert self.trueModel.rReps == 1
        assert mReps is not None
        self.mReps = mReps
        self.mReps_has_increased = False
        self.trueModel.extend_data(self.mReps)
        self.history = [('m', self.mReps)]

    def extend_m(self, m_delta):
        self.mReps += m_delta
        self.trueModel.extend_data(self.mReps)
        self.history.append(('m+', m_delta))
        self.mReps_has_increased = True

    def extend_k(self, k_delta):
        # X new likelihood ratios of size k_delta x r
        # X generate sum table
        # X generate sums and numbers vectors
        # sums gets appended, numbers gets added
        k_new = k_delta + len(self.sums_kx1)
        r = self.numbers_1xr.shape[1]
        new_sums, new_numbers = self.new_margins(k_delta, r)
        self.sums_kx1 = np.append(self.sums_kx1, new_sums, axis=0)
        self.numbers_1xr = self.numbers_1xr + new_numbers  # summands are type numpy.matrix, so + adds matrices
        assert self.sums_kx1.shape == (k_new, 1)
        assert self.numbers_1xr.shape == (1, r)
        self.history.append(('k+', k_delta))

    def extend_r(self, r_delta):
        # X new likelihood ratios of size k x rdelta
        # X generate sum table with previous cumsum added in
        # generate sums and numbers vectors
        # sums gets replaced, numbers gets appended
        k = self.sums_kx1.shape[0]
        r_new = r_delta + self.numbers_1xr.shape[1]
        new_sums, new_numbers = self.new_margins(k, r_delta, self.sums_kx1)
        self.sums_kx1 = new_sums
        self.numbers_1xr = np.append(self.numbers_1xr, new_numbers, axis=1)
        assert self.sums_kx1.shape == (k,1)
        assert self.numbers_1xr.shape == (1, r_new)
        self.history.append(('r+', r_delta))

    def root_table(self, k=1, r=1, mReps=None):
        if mReps is not None:
            self.mReps = mReps
            self.trueModel.extend_data(self.mReps)
        self.sums_kx1, self.numbers_1xr = self.new_margins(k, r)
        self.mReps_has_increased = False
        self.history.append(('k, r, m', (k, r, self.mReps)))

    def new_margins(self, k, r, old_sums=None):
        lr = self.new_likelihood_ratios(k, r)
        cs = self.new_cumsum_table(lr, old_sums)
        new_sums = cs[:, -1]
        new_numbers = np.sum(cs > 0, axis=0)
        return new_sums, new_numbers

    def new_likelihood_ratios(self, k, r):
        selection = toy.Select(self.trueModel, k*r, self.mReps, seed_or_state=False, RNG=self.RNG)
        M = self.hyp.likeRatios(self.alt, self.trueModel, selection)
        return M.reshape(k,r)

    def new_cumsum_table(self, LRs, old_sums=None):
        cumsums = np.cumsum(LRs, axis=1)
        return cumsums if old_sums is None else cumsums + old_sums

    def proportions(self):
        return self.numbers_1xr/float(self.sums_kx1.shape[0])

    def endpoints_index(self, C=0.95):
        P = self.proportions()
        i_min = 0  # first at or above confidence level
        while i_min < P.shape[1] and P[0, i_min] < C:
            i_min += 1
        i_max = P.shape[1] - 1  # last at or below confidence level
        while i_max >= 0 and P[0, i_max] > C:
            i_max -= 1
        return i_min, i_max

    def endpoints_repetitions(self, C=0.95):
        i_min, i_max = self.endpoints_index(C)
        return i_min+1, i_max+1  # repetitions start at 1; indexes start at 0

    def width(self, C=0.95):
        i_min, i_max = self.endpoints_index(C)
        return i_max-i_min  # could be -1

    def fitting_region(self, C=0.95):
        i_min, i_max = self.endpoints_index(C)
        w = i_max-i_min
        if w > 0:
            return range(i_min-w, i_max+w+1)
        else:
            assert w == -1 or w == 0
            return range(i_max, i_min+1)

    def fitting_xy(self, C=.95):
        region = self.fitting_region(C)
        P = self.proportions()
        return np.matrix(region)+1, P[:,region]

    def regression_line(self, C=.95):
        x, y = self.fitting_xy(C)
        mb = np.polyfit(np.array(x).ravel(), np.array(y).ravel(), 1)
        return mb

    def r_star(self, C=.95):
        if self.all_above(C):
            return 1.
        else:
            m, b = self.regression_line(C)
            return float(C - b)/float(m)

    def all_below(self, C=.95):
        r_total = self.numbers_1xr.shape[1]
        r_min, r_max = self.endpoints_repetitions(C)
        return r_min == r_total+1 and r_max == r_total

    def all_above(self, C=0.95):
        r_min, r_max = self.endpoints_repetitions(C)
        return r_min == 1 and r_max == 0

    def last_below(self, C=.95):
        return self.numbers_1xr[0,-1]/float(self.sums_kx1.shape[0]) < C

    def need_r(self, C=0.95):
        i_min, i_max = self.endpoints_index()
        if self.last_below():
            return True
        else:
            return i_max + 2*self.width() > self.numbers_1xr.shape[1]

    def report(self, C=.95):
        r_total = self.numbers_1xr.shape[1]
        r_min, r_max = self.endpoints_repetitions(C)
        print "Confidence Level:", C
        if r_min == r_total+1 and r_max == r_total:
            print "All repetitions to", r_total, 'are below confidence threshold'
        elif r_min == 1 and r_max == 0:
            print "All repetitions to", r_total, 'are above confidence threshold'
        else:
            print "(first_at_or_above, last_at_or_below) = (", r_min, ",", r_max, ") up to",\
                  r_total,"repetitions"
        print "Each repetition derived from a sample of", self.sums_kx1.shape[0], "bootstrapped likelihoods"
        print "Total of", r_total*self.sums_kx1.shape[0], "Bootstrapped Likelihoods."
        print "Bootstrapping from a Monte Carlo sample size of:", self.mReps
        if self.mReps_has_increased:
            print "The Monte Carlo sample size has increased since the table was rooted."