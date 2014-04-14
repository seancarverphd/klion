import numpy
import parameter

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
        self.reorder()       # defines nodeOrder dictionary
        self.disconnect()  #defines self.PS and self.Q = zero-matrix
        self.integrity()
    def reorder(self):
        self.nodeOrder = {}
        for n in range(len(self.nodes)):
            self.nodeOrder[self.nodes[n].name] = n
    def getLevels(self):
        levels = []
        for n in self.nodes:
            levels.append(n.level)
        return set(levels)
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
    def addNode(self,new):
        assert(False) # haven't finished coding, need to add row and column to QList and Q
        self.nodes.append(new)
        self.PS.append(new.level.PS)
        self.reorder()
    #The next four functions define/modify the Q matrix
    #disconnect() defines a disconnected graph; no transitions
    def disconnect(self):
        self.Q = numpy.matrix(numpy.zeros(shape=(len(self.nodes),len(self.nodes))))
        self.QList = self.Q.tolist()   # makeQ makes Q from QList, here the other way around
        self.PS = parameter.emptySpace()  # clear the parameter space
        for n in self.nodes:
            self.PS.append(n.level.PS)          # now recreate the parameter space from just nodes
    #fillQdiag() enforces (by modifying the diagonal of Q): sum of each row is zero
    def makeQ(self):
        self.Q = numpy.matrix(self.QList)
        Qdiag = -self.Q.sum(axis=1)
        numpy.fill_diagonal(self.Q,Qdiag)
    #addBiEdge() modifies parameters of a transition in both directions
    def addBiEdge(self,first,second,q12,q21):
        self.QList[first][second] = q12 # first row, second column, order reverse in list
        self.QList[second][first] = q21 # second row, first column
        self.PS.append(parameter.getSpace(q12))
        self.PS.append(parameter.getSpace(q21))
        self.integrity()  # calls makeQ()
    #addEdge() modifies parameters of a transition in one direction
    def addEdge(self,first,second,q12):
        self.QList[first][second] = q12
        self.PS.append(parameter.getSpace(q12))
        self.integrity() # calls makeQ()
    def integrity(self): # Checks that channel is well defined
        #Nodes
        for n in self.nodes:
            assert(isinstance(n,Node))
        #EdgesQ
        for n in range(len(self.nodes)):
            assert(self.QList[n][n]==0)
        self.makeQ()
        Q0 = self.Q.copy()
        numpy.fill_diagonal(Q0,0.)
        assert(numpy.amin(Q0)==0)
        assert(self.Q.shape == (len(self.nodes),len(self.nodes)))

#This code sets up a canonical channel
Open = Level("Open",1.0,0.6)
Closed = Level("Closed",0.0,0.3)
C1 = Node("C1",Closed)
C0 = Node("C0",Closed)
O = Node("O",Open)
ch3 = Channel([C1,C0,O])
ch3.addBiEdge(0,1,2.,3.)
ch3.addEdge(1,2,4.)
ch3.addEdge(2,1,5.)