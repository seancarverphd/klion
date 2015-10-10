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

    def endpoints(self, C=0.95):
        P = self.proportions()
        r_min = 0
        while r_min < P.shape[1] and P[0, r_min] < C:
            r_min += 1
        r_max = P.shape[1] - 1
        while r_max >= 0 and P[0, r_max] > C:
            r_max -= 1
        return r_min, r_max

    def width(self, C=0.95):
        r_min, r_max = self.endpoints(C)
        return r_max - r_min

    def report(self, C=.95):
        r_total = self.numbers_1xr.shape[1]
        r_min, r_max = self.endpoints(C)
        print "Confidence Level:", C
        if r_min == r_total and r_max == r_total-1:
            print "All repetitions to", r_total, 'are below confidence threshold'
        elif r_min == 0 and r_max == -1:
            print "All repetitions to", r_total, 'are above confidence threshold'
        else:
        print "(first_at_or_above, last_at_or_below) = (", r_min+1, ",", r_max+1, ") up to",\
                r_total,"repetitions"
        print "Each repetition derived from a sample of", self.sums_kx1.shape[0], "bootstrapped likelihoods"
        print "Total of", r_total*self.sums_kx1.shape[0], "Bootstrapped Likelihoods."
        print "Bootstrapping from a Monte Carlo sample size of:", self.mReps
        if self.mReps_has_increased:
            print "The Monte Carlo sample size has increased since the table was rooted."