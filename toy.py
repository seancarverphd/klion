import numpy
import random
import time

class toyProtocol(object):
    def __init__(self, q, seed=None):
        self.R = random.Random()
        self.seed = seed
        self.clearData()
        self.changeModel(q)
    def clearData(self):
        self.nReps = None
        if self.seed == None:
            self.usedSeed = long(time.time()*256)
        else:
            self.usedSeed = self.seed  # can be changed with self.reseed()
        self.R.seed(self.usedSeed)  # For simulating Markov Chain
        self.taus = []
    def reseed(self,seed):
        self.seed = seed
        self.clearData()  # Reseeds random number generator
    def changeModel(self,q):
        if len(q) == 1:
            self.toy2 = True
            self.q = q[0]
        elif len(q) == 2:
            self.toy2 = False
            self.q0 = q[0]
            self.q1 = q[1]
        else:
            assert(False) # Length of q should be 1 or 2
    def sim(self,nReps=1,clear=False): # Only does new reps; keeps old; if (nReps < # Trajs) then does nothing
        if clear:
            self.clearData()  # Reseeds random number generator
        numNewReps = nReps - len(self.taus)
        for n in range(numNewReps):  
            if self.toy2:
                self.taus.append(self.R.expovariate(self.q))
            else:
                self.taus.append(self.R.expovariate(self.q1)+self.R.expovariate(self.q0))
        self.nReps = nReps
    def minuslike2(self):      
        self.mll = 0.
        for n in range(self.nReps):
            self.mll -= numpy.log(self.q) - self.q*self.taus[n]
        return self.mll
    def minuslike3(self):
        # q1 must be different from q0 otherwise get 0/0
        self.mll = 0.
        for n in range(self.nReps):
            self.mll -= numpy.log(self.q1) + numpy.log(self.q0)
            self.mll -= numpy.log((numpy.exp(-self.q0*self.taus[n])-numpy.exp(-self.q1*self.taus[n]))/(self.q1-self.q0))
        return self.mll
    def minuslike(self):
        if self.toy2:
            return self.minuslike2()
        else:
            return self.minuslike3()
    def like2(self):
        return -self.minuslike2()
    def like3(self):
        return -self.minuslike3()
    def like(self):
        return -self.minuslike()
    
T = toyProtocol([.5,.25])
T.sim(nReps=10)
print T.like()