import numpy
import matplotlib
import matplotlib.pylab as plt
import parameter
import toy
import cPickle as pickle
import csv
import scipy.stats as stats

def Ksave(K,fname):
    f = open(fname,'wb')
    pickle.dump(K.qRange,f)
    pickle.dump(K.KL,f)
    f.close()
    
def Kload(fname):
    f = open(fname,'rb')
    qRange = pickle.load(f)
    KL = pickle.load(f)
    f.close()
    K = Kshell(qRange,KL)
    return K

def Psave(P,fname):
    f = open(fname,'wb')
    pickle.dump(P.prop,f)
    f.close
    
def Pload(fname):
    f = open(fname,'rb')
    prop = pickle.load(f)
    f.close()
    P = Pshell(prop)
    return P

class Kshell(object):  # better name: Kskeleton
    def __init__(self,qRange,KL):
        self.qRange = qRange
        self.KL = KL
    def plot(self):
        cs = plt.pcolor(self.qRange,self.qRange,self.KL)
        cb = plt.colorbar(cs)
        plt.title("$D_{KL}$ As True Model's Parameters Vary")
        plt.xlabel('First Rate Constant (kHz)')
        plt.ylabel('Second Rate Constant (kHz)')
        # Should save nReps and derive '10^5' from nReps=100000
        cb.set_label('$D_{KL}$ to Closest 2-State Alternative\nBased on $10^5$ Monte Carlo Samples')
        plt.show()

class Pshell(object):
    def __init__(self,prop):
        self.prop = prop
    def plot(self):
        ax = plt.gca()
        reject = matplotlib.patches.Rectangle((0,75),50,20,color='red',alpha=.3)
        accept = matplotlib.patches.Rectangle((0,95),50,5,color='green',alpha=.3)
        ax.add_patch(reject)
        ax.add_patch(accept)
        plt.plot(range(1,51),numpy.array(self.prop)*100,'*')
        plt.xlabel("Number of Independent Repetitions of Experiment (N)")
        plt.ylabel("P(True Model Selected), ($P_N$, Percent)")
        plt.text(.5,99,"Selection Correct Sufficiently Often")
        plt.text(.5,94,"Selection Correct Insufficiently Often")
        plt.show()
    
class kull(object):
    def __init__(self,trueParent,altParent,q0=None,q1=None,q=None):
        self.q0 = q0
        self.q1 = q1
        self.q = q
        self.trueParent = trueParent
        self.altParent = altParent
        self.TrueMod = trueParent.flatten()
        self.AltMod = altParent.flatten()
        self.qRange = numpy.arange(.1,10,.1) # use for plotting
        #self.qRange = numpy.arange(1,10,2)  # use for testing
        self.nReps = 100000
    def compute(self):
        num = len(self.qRange)
        self.KL = numpy.zeros((num,num))
        for i,q_1 in enumerate(self.qRange):
            for j,q_0 in enumerate(self.qRange):
                self.q0.assign(q_0)
                self.q1.assign(q_1)
                self.TrueMod.changeModel(self.trueParent)
                self.TrueMod.sim(self.nReps,clear=True)
                EfTrue = self.TrueMod.Eflogf()
                mEfAlt = numpy.log(numpy.e*numpy.mean(self.TrueMod.taus))  # Simple because toy is simple
                self.KL[i,j] = EfTrue + mEfAlt
                print i, j, "up to", len(self.qRange)
    def save(self,fname):
        Ksave(self,fname)

class FBplot(object):
    def __init__(self,trueParent,altParent,q0=None,q1=None,q=None):
        self.q0 = q0
        self.q1 = q1
        self.q = q
        self.trueParent = trueParent
        self.altParent = altParent
        self.TrueMod = trueParent.flatten(50) # setting seed
        self.AltMod = altParent.flatten(51) # setting seed
        self.nReps = 100000
    def compute(self):
        self.TrueMod.sim(self.nReps)
        self.aa = self.TrueMod.aic(self.AltMod)
        self.caa = numpy.cumsum(self.aa)
        self.abar = []
        self.amabar2 = []
        for n,ca in enumerate(self.caa.tolist()[0]):
            self.abar.append(ca/(n+1))  # don't want to divide by 0
            self.amabar2.append((self.aa[0,n]-self.abar[n])**2)
        self.cam = numpy.cumsum(self.amabar2)
        self.vm = []
        for m,ca in enumerate(self.cam.tolist()):
            self.vm.append(ca/((m+.0001)**2))  # don't want to divide by 0 except dividing by m-1
    def plot(self):
        n = range(1,len(self.abar)+1)  # don't want to take log of 0
        n2 = range(2,len(self.abar)+1)
        plt.plot(numpy.log10(n),numpy.array(self.abar)/2.)
        self.arraylow = numpy.array(self.abar-numpy.sqrt(self.vm))
        self.arrayhigh = numpy.array(self.abar+numpy.sqrt(self.vm))
        self.arraylow[0] = -100
        self.arrayhigh[0] = 100
        plt.fill_between(numpy.array(numpy.log10(n2)),self.arraylow[1:]/2.,self.arrayhigh[1:]/2.,color='red',alpha=.5)
        ax = plt.gca()
        ax.set_axis_bgcolor('wheat')
        plt.axis([0,5,-.1,.35])
        plt.title("Estimating $D_{KL}$")
        plt.ylabel("$D_{KL}$ Estimate +/- One Standard Error")
        plt.xlabel("Log10 Size of Monte Carlo Sample")
        
class aichist(object):
    def __init__(self,trueParent,altParent,q0=None,q1=None,q=None):
        self.q0 = q0
        self.q1 = q1
        self.q = q
        self.trueParent = trueParent
        self.altParent = altParent
        self.TrueMod = trueParent.flatten(seed=554)
        self.AltMod = altParent.flatten(seed=555)
        self.nReps = 100000
    def compute(self):
        self.AIC = numpy.zeros(self.nReps)
        self.TrueMod.sim(self.nReps,clear=True)
        #fTrue = self.TrueMod.logf()
        #gAlt = self.AltMod.logf(self.TrueMod.taus[0:self.nReps])
        #self.AIC = 2.*(fTrue - gAlt)
        self.AIC = self.TrueMod.aic(self.AltMod)
    def plot(self):
        ax = plt.gca()
        reject = matplotlib.patches.Rectangle((-12,0),12,12,color='red',alpha=.3)
        accept = matplotlib.patches.Rectangle((0,0),2,12,color='green',alpha=.3)
        ax.add_patch(reject)
        ax.add_patch(accept)
        plt.hist(H.AIC.T,1000,normed=True,color='black')
        plt.axis([-3,1,0,12])
        plt.title('Histogram of AIC Differences')
        plt.xlabel('AIC Difference')
        plt.ylabel('Density')  
        mAIC = numpy.mean(H.AIC)
        plt.arrow(mAIC,3.5,0,-2.1, fc="k", ec="k", head_width=0.05, head_length=0.1)
        plt.text(mAIC-.1,3.75,"Mean \n= $2 D_{KL}$")
        plt.text(-2.5,11.4,"Alternative Incorrectly Selected")
        percenttrue = (numpy.sum(H.AIC>0)*100)/numpy.size(H.AIC)
        plt.text(-1.75,11,str(100-percenttrue)+'%')
        plt.text(.1,11.4,"True Selected")
        plt.text(.35,11,str(percenttrue)+'%')
        # props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
        # ax.text(-2.2,5,"Comparing 3-state true model:\n     $C_1$ --> $C_0$ --> $O$; $q_1$ = 1/4, $q_0$ = 1/2\n\nTo 2-state alternative:\n     $C$ --> $O$; $q$ = 1/6",bbox=props)
        plt.show()
    def save(self,fname):  # For excel or StatCrunch
        with open(fname,'w') as csvfile:
            writ = csv.writer(csvfile, delimiter='\n',quotechar='|',quoting=csv.QUOTE_MINIMAL)
            writ.writerow(numpy.array(self.AIC)[0].tolist())
            #for a in numpy.array(self.AIC)[0].tolist():
            #    writ.writerow(a)

class PNplot(object):
    def __init__(self,trueParent,altParent,q0=None,q1=None,q=None):
        self.q0 = q0
        self.q1 = q1
        self.q = q
        self.trueParent = trueParent
        self.altParent = altParent
        self.M = 100000
        self.initseed = 111
        self.rangePlot = 51
        self.alpha = 95
        self.AIC = []
        self.prop = []
        self.mn = None
        self.sd = None
    def compute(self):
        for N in range(1,self.rangePlot):
            self.TrueMod = self.trueParent.flatten(seed=self.initseed+2*N)
            self.AltMod = self.altParent.flatten(seed=self.initseed+2*N+1)
            self.AIC.append(self.TrueMod.aicN(self.AltMod,self.M,N))
            self.prop.append(numpy.sum(self.AIC[-1]>0)/float(self.M))
    def theoretical(self):
        self.TrueMod = self.trueParent.flatten()
        self.AltMod = self.altParent.flatten()
        self.TrueMod.sim(nReps=self.M)
        (self.mn,self.sd) = self.TrueMod.a_mn_sd(self.AltMod)
        self.nRange = range(1,self.rangePlot)
        self.PNtheo = [] 
        for n in self.nRange:
            self.PNtheo.append(stats.norm.cdf(numpy.sqrt(n)*self.mn/self.sd))
    def theo2(self,PN):
        self.TrueMod = self.trueParent.flatten()
        self.AltMod = self.altParent.flatten()
        self.PN100 = numpy.array(self.PNtheo)*100
        self.Nfirst = 1+min(numpy.where(self.PN100-self.alpha>0)[0])  # add 1 cause index starts at 1
        self.PNfirst = PN.prop[self.Nfirst-1]
        self.rcv = stats.norm.ppf(self.PNfirst)/numpy.sqrt(self.Nfirst)
        self.PNtheo2 = []
        for n in self.nRange:
            self.PNtheo2.append(stats.norm.cdf(numpy.sqrt(n)*self.rcv))
    def save(self,fname):
        Psave(self,fname)
        
class Nalpha(object):
    def __init__(self,trueParent,altParent,q0=None,q1=None,q=None):
        self.q0 = q0
        self.q1 = q1
        self.q = q
        self.trueParent = trueParent
        self.altParent = altParent
        self.trueModel = trueParent.flatten()
        self.altModel = altParent.flatten()
        #self.qRange = numpy.arange(.1,10,.1) # use for plotting
        self.qRange = numpy.arange(1,10,2) # use for testing
        self.MCSampSize = 1000
        self.numWrongTarget = 50
        self.numRightTarget = self.MCSampSize - self.NumWrongTarget # 1000-50==950
        self.alpha = self.numWrongTarget/self.MCSampSize # 50/1000 == 0.05
    def compute(self):
        num = len(self.qRange)
        self.NA = numpy.zeros((num,num))
        for i,q_1 in enumerate(self.qRange):
            for j,q_0 in enumerate(self.qRange):
                self.q0.assign(q_0)
                self.q1.assign(q_1)
                # Only do trueModel once per nRep step
                self.nReps = max_nReps
                # nReps set adaptively to achieve alpha 
                self.trueModel.changeModel(self.trueParent)
                self.trueModel.sim4aic(self.nReps,self.MCSampSize)
                # binary search
                while (not numRight==self.numRightTarget) and (nRep <= max_nReps and nReps >= 1):
                    # self.altModel.sim4aic(self.nReps,self.MCSampSize)
                    if numRight < self.numRightTarget:
                        nReps = int(floor(nReps*1.33))
                    elif numRight > self.numRightTarget:
                        nReps = int(floor(nReps/2))
                    # More to come ...
                # More to come ...
    def save(self,fname):
        NAsave(self,fname)
        
q0 = parameter.Parameter("q0",0.5,"kHz",log=True)
q1 = parameter.Parameter("q1",0.25,"kHz",log=True)
q = parameter.Parameter("q",1./6.,"kHz",log=True)
T3 = toy.toyProtocol([q0,q1])
T2 = toy.toyProtocol([q])
H = aichist(T3,T2,q0,q1,q)
H.compute()
# print H.AIC

K = kull(T3,T2,q0,q1,q)
# Uncomment next three lines to generate 'largeKLDataSet.p' (untested); ToDo: Change code to save (& print in plot) nReps, and test
# K.compute()
# K.save('largeKLDataSet.p')
# print K.KL

P = PNplot(T3,T2,q0,q1,q)
# Uncomment next three lines to generate 'largePropDataSet' (untested);
# P.compute()
# P.save('largePNDataSet.p')

FB = FBplot(T3,T2,q0,q1,q)

# TO PRINT FIGURE 1
plt.figure(1)
KP = Kload('largeKLDataSet.p')  # This file not in repository; uncomment three lines above to generate; (needs testing)
KP.plot()

# TO PRINT FIGURE 2
plt.figure(2)
H.plot()

# TO PRINT FIGURE 3
plt.figure(3)
PN = Pload('largePNDataSet.p')
PN.plot()
P.theoretical()
v = plt.axis()
plt.plot(range(1,51),numpy.array(P.PNtheo)*100,'r',alpha=.6)
plt.axis(v)
plt.show()
P.theo2(PN)
plt.plot(range(1,51),numpy.array(P.PNtheo2)*100,'r',alpha=.5)

# TO PRINT FIGURE 4
plt.figure(4)
FB.compute()
FB.plot()
plt.show()
