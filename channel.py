import numpy
import parameter
from parameter import u

# A Level is a defined (mean, std) that will be the same across certain Nodes (ie states)
class Level(object):
    def __init__(self,name,mean,std):
        self.name = name
        self.mean = mean
        self.std = std
        self.integrity() # calls reparameterize()
    def reparameterize(self):
        self.PS = parameter.emptySpace()
        self.PS.append(parameter.getSpace(self.mean))
        self.PS.append(parameter.getSpace(self.std))
        # called by integrity()
    def setMean(self,mean):
        self.mean = mean
        self.integrity() # calls reparameterize()
    def setStd(self,std):
        self.std = std
        self.integrity() # calls reparameterize()
    def __repr__(self):
        return '%s\n  Mean %s\n  Std %s' % (self.name,repr(self.mean),repr(self.std))
    def __str__(self):
        return '%s\n  Mean %s\n  Std %s' % (self.name,str(self.mean),str(self.std))
    def integrity(self):
        self.reparameterize()
        assert(isinstance(self.name,basestring))
        assert float(self.std.Value()) >= 0.
        
# A Node is a state of the channel
class Node(object):
    def __init__(self,name,level):
        self.name = name
        self.level = level
        self.integrity()
    def __repr__(self):
        return 'Node %s: %s' % (self.name, repr(self.level))
    def __str__(self):
        return 'Node %s: %s' % (self.name, self.level.name)
    def integrity(self):
        assert(isinstance(self.name,basestring))
        assert(isinstance(self.level,Level))
        
# A Channel is a model of an ion channel
class Channel(object):
    def __init__(self,nodes):
        # Define dictionary of nodes
        self.nodes = nodes
        self.recordOrder() # defines nodeOrder dictionary
        self.disconnect()  # sets Q and QList = zero-matrix initializes; calls integrity() which calls reparametrize()
    def makeQ(self):
        # make a QList without the units
        flatQ = []
        for row in self.QList:
            flatrow = []
            for element in row:
                try:
                    flatrow.append(float(element))
                except:
                    flatrow.append(element.evaluate())
            flatQ.append(flatrow)
        # Convert to matrix
        Q = numpy.matrix(flatQ)
        # Add diagonal (not zero)
        Qdiag = -Q.sum(axis=1)
        numpy.fill_diagonal(Q,Qdiag)
        return Q
    def makeM(self):
        means = []
        for n in self.nodes:
            try:
                means.append(float(n.level.mean))
            except:
                means.append(n.level.mean._magnitude)
        return means
    def recordOrder(self):
    # records order of nodes into a dictionary to reference them by string name
        self.nodeOrder = {}
        for n in range(len(self.nodes)):
            self.nodeOrder[self.nodes[n].name] = n
    def getLevels(self):
        levels = []
        for n in self.nodes:
            levels.append(n.level)
        return set(levels)
    def getNodeNames(self):
        nodeNames = []
        for n in self.nodes:
            nodeNames.append(n.name)
        return set(nodeNames)  # should all be distinct, used for checking distictiveness
    def __repr__(self):
        nNodes = len(self.nodes)
        s = 'Channel'
        for l in self.getLevels():
            s += '\n Level: '+str(l)
        for n in self.nodes:
            s += '\n '
            s += str(n)
        for i in range(0, nNodes-1):
            for j in range(i+1, nNodes):
                if self.QList[i][j] == 0. and self.QList[j][i] == 0.:
                    assert(True)
                elif self.QList[j][i] == 0.:
                    s += '\n Edge %s --> %s:\n q (-->) %s' % (self.nodes[i].name, self.nodes[j].name, str(self.QList[i][j]))
                elif self.QList[i][j] == 0:
                    s += '\n Edge %s <-- %s:\n q (<--) %s' % (self.nodes[i].name, self.nodes[j].name, str(self.QList[j][i]))
                else:
                    s += '\n Edge %s <--> %s:\n  q (-->) %s\n  q (<--) %s' % (self.nodes[i].name, self.nodes[j].name, str(self.QList[i][j]),str(self.QList[j][i]))
        s += '\n'+str(self.PS)
        return s
    def padQList(self):
    # Add a new row and column to QList
        newrow = []
        for row in self.QList:
            row.append(0)  # adds new column element by element
            newrow.append(0) # adds final column
        newrow.append(0)
        self.QList.append(newrow)
    def addNode(self,new):
        self.nodes.append(new)
        self.PS.append(new.level.PS)
        self.recordOrder()
        self.padQList()
        self.integrity()
    #The next three functions define/modify the edges
    def disconnect(self):
    #disconnect() defines a disconnected graph; no transitions
        self.QList = numpy.matrix(numpy.zeros(shape=(len(self.nodes),len(self.nodes)))).tolist()
        self.integrity()  # calls makeQ() and reparameterize()
    def biEdge(self,node1,node2,q12,q21):
    #addBiEdge() modifies parameters of a transition in both directions
        first = self.nodeOrder[node1]
        second = self.nodeOrder[node2]
        self.QList[first][second] = q12 # first row, second column, order reverse in list
        self.QList[second][first] = q21 # second row, first column
        self.integrity()  # calls makeQ() and reparameterize()
    def edge(self,node1,node2,q12):
    #addEdge() modifies parameters of a transition in one direction
        first = self.nodeOrder[node1]
        second = self.nodeOrder[node2]
        self.QList[first][second] = q12
        self.integrity() # calls makeQ() and reparameterize()
    def reparameterize(self):
    # defines parameter space;  called by integrity()
        self.PS = parameter.emptySpace()  # clear the parameter space
        for n in self.nodes:
            self.PS.append(n.level.PS)          # now recreate the parameter space from just nodes
        for rownum in range(len(self.QList)):
            for element in self.QList[rownum]:
                self.PS.append(parameter.getSpace(element))
    def integrity(self): # Checks that channel is well defined
        #Nodes
        for n in self.nodes:
            assert(isinstance(n,Node))
        assert(len(self.nodes) == len(self.getNodeNames())) # makes sure node names are distinct 
        assert(len(self.nodes) == len(set(self.nodes)))  # make sure nodes are distinct
        #Edges
        for n in range(len(self.nodes)):
            assert(self.QList[n][n]==0)
        #Q0 = self.Q.copy()  # Q0 is for checking that off diagonal is positive
        #numpy.fill_diagonal(Q0,0.)  # diagonal is negative so set to zero
        #assert(numpy.amin(Q0)==0)  # now minimum element should be zero (on diagonal)
        #assert(self.Q.shape == (len(self.nodes),len(self.nodes)))
        self.reparameterize()

#This code sets up a canonical channel
gmax_khh = parameter.Parameter("gmax_khh",0.02979,"microsiemens",log=True)
gstd_open = parameter.Parameter("gstd_open", 0.01,"microsiemens",log=True)
gstd_closed = parameter.Parameter("gstd_closed",0.001,"microsiemens",log=True)
# for one channel units of gmax_khh should be siemens
ta1 = parameter.Parameter("ta1",4.4,"ms",log=True)
tk1 = parameter.Parameter("tk1",-0.025,"1/mV",log=False)
d1 = parameter.Parameter("d1",21.,"mV",log=False)
k1 = parameter.Parameter("k1",0.2,"1/mV",log=False)

ta2 = parameter.Parameter("ta2",2.6,"ms",log=True)
tk2 = parameter.Parameter("tk2",-0.007,"1/mV",log=False)
d2 = parameter.Parameter("d2",43,"mV",log=False)
k2 = parameter.Parameter("k2",0.036,"1/mV",log=False)

v = parameter.Parameter("v",-65.,"mV",log=False)
vr = parameter.Expression("vr","v + 65*u.mV",[v])
tau1 = parameter.Expression("tau1","ta1*exp(tk1*vr)",[ta1,tk1,vr])
K1 = parameter.Expression("K1","exp((k2*(d2-vr))-(k1*(d1-vr)))",[k1,k2,d1,d2,vr])
tau2 = parameter.Expression("tau2","ta2*exp(tk2*vr)",[ta2,tk2,vr])
K2 = parameter.Expression("K2","exp(-(k2*(d2-vr)))",[k2,d2,vr])

a1 = parameter.Expression("a1","K1/(tau1*(K1+1))",[K1,tau1])
b1 = parameter.Expression("b1","1/(tau1*(K1+1))",[K1,tau1])
a2 = parameter.Expression("a2","K2/(tau2*(K2+1))",[K2,tau2])
b2 = parameter.Expression("b2","1/(tau2*(K2+1))",[K2,tau2])

Open = Level("Open",mean=gmax_khh,std=gstd_open)
Closed = Level("Closed",mean=0.*u.microsiemens,std=gstd_closed)
C1 = Node("C1",Closed)
C2 = Node("C2",Closed)
O = Node("O",Open)
khh = Channel([C1,C2,O])
khh.biEdge("C1","C2",a1,b1)
khh.edge("C2","O",a2)
khh.edge("O","C2",b2)