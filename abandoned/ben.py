import numpy
import patch
import channel
from parameter import u
import parameter
import engine
import matplotlib.pyplot as plt

# Comparing time for the following two
# functions shows that appending is
# slightly faster than inserting, at
# least on my machine.  Unlike MATLAB.

def loop_append(n):
    list = []
    for i in range(n):
        list.append(i)
    return list

def loop_insert(n):
    list = numpy.zeros(n)
    for i in range(n):
        list[i] = i
    return list

def loop_matrix(n):
    list = numpy.matrix(numpy.zeros(n))
    for i in range(n):
        list[0,i] = i
    return list

P = patch.singleChannelPatch(channel.khh)
#voltages = [channel.V0,channel.V1,channel.V2,channel.V1]  # repeat V1; repeated variables affect differentiation via chain rule
#voltageStepDurations = [0*u.ms,patch.default_tstop,patch.default_tstop,patch.default_tstop]  # default_tstop is a global parameter
voltages = [channel.V0,channel.V1,channel.V0,channel.V2]
voltageStepDurations = [numpy.inf,patch.default_tstop,numpy.inf,patch.default_tstop]
S = patch.StepProtocol(P,voltages,voltageStepDurations)
FS = S.flatten(3)
FS.sim(1)
FS.simG(1)
DF = FS.dataFrame()
# RS = RepeatedSteps(P,voltages,voltageStepDurations)
# RS.sim(rng=3,nReps=4)
plt.plot(DF.T_ms,DF.G_pS)
plt.show()
