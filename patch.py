import channel
import numpy as np

def equilQ(Q):
    (V,D) = np.linalg.eig(Q.T)   # eigenspace
    imin = np.argmin(np.absolute(V))  # index of 0 eigenvalue
    eigvect0 = D[:,imin]  # corresponding eigenvector
    return eigvect0.T/sum(eigvect0) # normalize and return (fixes sign)
class Patch(object):
    def __init__(self, channels):
        self.channels = channels
        self.assertOneChannel()
        self.Q = self.ch.makeQ()
        self.Mean = self.ch.makeMean()
        self.Std = self.ch.makeStd()
    def assertOneChannel(self):
        assert(len(self.channels) == 1)
        assert(self.channels[0][0] == 1)
        self.ch = self.channels[0][1]
        assert(isinstance(self.ch,channel.Channel))
    def equilibrium(self):
        self.Q = self.ch.makeQ()
        return equilQ(self.Q)

P = Patch([(1, channel.khh)])