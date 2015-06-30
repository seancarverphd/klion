import engine
import parameter
import toy

__author__ = 'sean'

preferred = parameter.preferredUnits()
preferred.time = 'ms'
preferred.freq = 'kHz'


class ToyEnsemble(object):
    def __init__(self, q=None, q0=None, q1=None,
                 n2=None, n3=None, dt=None, tstop=None):
        self.preferred = preferred
        if n2 > 0:
            assert q is not None
            self.q = q
        if n3 > 0:
            assert q0 is not None
            assert q1 is not None
            self.q0 = q0
            self.q1 = q1
        self.n2 = int(n2)
        self.n3 = int(n3)
        self.dt = dt
        self.tstop = tstop

    def flatten(self, seed=None):
        parent = self  # for readability
        flat = FlatToyEnsemble(parent, seed)
        return flat


class FlatToyEnsemble(engine.FlatStepProtocol):
    def initRNG(self, seed=None):
        return toy.SaveStateRNG(seed)

    def setUpExperiment(self, parent):
        self.preferred_time = parent.preferred.time
        self.preferred_freq = parent.preferred.freq
        self.q = parameter.mu(parent.q, self.preferred_freq)
        self.q0 = parameter.mu(parent.q0, self.preferred_freq)
        self.q1 = parameter.mu(parent.q1, self.preferred_freq)
        self.dt = parameter.mu(parent.dt, self.preferred_time)
        self.n2 = int(parent.n2)
        self.n3 = int(parent.n3)
        self.tstop = parameter.mu(parent.tstop, self.preferred_time)
        self.nsamples = int(self.tstop/self.dt)
        self.processNodes(None) # parameter for consistency with super class

    def processNodes(self, nodes):
        assert nodes is None
        pass
