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
        return '%s\n Mean %s\n Std %s' % (self.name,str(self.mean),str(self.std))
    def integrity(self):
        self.reparameterize()
        assert(isinstance(self.name,basestring))
        float(self.mean)
        assert float(self.std) >= 0.
        
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
            s += '\n Level: '+repr(l)
        for n in self.nodes:
            s += '\n '
            s += str(n)
        for i in range(0, nNodes-1):
            for j in range(i+1, nNodes):
                if self.Q[i,j] == 0. and self.Q[j,i] == 0.:
                    assert(True)
                elif self.Q[j,i] == 0.:
                    s += '\n Edge %s --> %s:\n q (-->) %s' % (self.nodes[i].name, self.nodes[j].name, str(self.Q[i,j]))
                elif self.Q[i,j] == 0:
                    s += '\n Edge %s <-- %s:\n q (<--) %s' % (self.nodes[i].name, self.nodes[j].name, str(self.Q[j,i]))
                else:
                    s += '\n Edge %s <--> %s:\n  q (-->) %s\n  q (<--) %s' % (self.nodes[i].name, self.nodes[j].name, str(self.Q[i,j]),str(self.Q[j,i]))
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
    #The next four functions define/modify the Q matrix
    def disconnect(self):
    #disconnect() defines a disconnected graph; no transitions
        self.Q = numpy.matrix(numpy.zeros(shape=(len(self.nodes),len(self.nodes))))
        self.QList = self.Q.tolist()   # makeQ makes Q from QList, here the other way around
        self.integrity()  # calls makeQ() and reparameterize()
    def makeQ(self):
    #fillQdiag() enforces (by modifying the diagonal of Q): sum of each row is zero
        self.Q = numpy.matrix(self.QList)
        Qdiag = -self.Q.sum(axis=1)
        numpy.fill_diagonal(self.Q,Qdiag)
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
        self.makeQ()
        Q0 = self.Q.copy()  # Q0 is for checking that off diagonal is positive
        numpy.fill_diagonal(Q0,0.)  # diagonal is negative so set to zero
        assert(numpy.amin(Q0)==0)  # now minimum element should be zero (on diagonal)
        assert(self.Q.shape == (len(self.nodes),len(self.nodes)))
        self.reparameterize()

#This code sets up a canonical channel
gmax_khh = parameter.Parameter("gmax_khh",0.02979,"siemens/cm**2",log=True)
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
vr = parameter.Expression("v + 65*u.mV",[v])
tau1 = parameter.Expression("ta1*exp(tk1*vr)",[ta1,tk1,vr])

Open = Level("Open",mean=1.0,std=0.6)
Closed = Level("Closed",mean=0.0,std=0.3)
C1 = Node("C1",Closed)
C0 = Node("C0",Closed)
O = Node("O",Open)
ch3 = Channel([C1,C0,O])
ch3.biEdge("C1","C0",2.,3.)
ch3.edge("C0","O",4.)
ch3.edge("O","C0",5.)