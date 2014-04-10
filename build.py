import numpy

# A Level is a defined (mean, std) that will be the same across certain Nodes (ie states)
class Level(object):
    def __init__(self,name,mean,std):
        assert(isinstance(name,basestring))
        mean = float(mean)
        std = float(std)
        self.name = name
        self.mean = mean
        self.std = std
    def __repr__(self):
        return 'Level %s: (mean %s, std %s)' % (self.name,repr(self.mean),repr(self.std))

# A Node is a state of the channel
class Node(object):
    def __init__(self,name,level):
        assert(isinstance(name,basestring))
        assert(isinstance(level,Level))
        self.name = name
        self.level = level
    def __repr__(self):
        return 'Node %s: %s' % (self.name, repr(self.level))

# A Channel is a model of an ion channel
class Channel(object):
    def __init__(self,nodes):
        for node in nodes:
            assert(isinstance(node,Node))
        self.nodes = nodes
        self.disconnect()  #defines self.Q = matrix[zero]
    def __repr__(self):
        s = 'Channel with Nodes:'
        for n in self.nodes:
            s += '\n'
            s += repr(n)
        return s
    def addNode(self,new):
        self.nodes.append(new)
    #The next four functions define/modify the Q matrix
    #disconnect() defines a disconnected graph; no transitions
    def disconnect(self):
        self.Q = numpy.matrix(numpy.zeros(shape=(len(self.nodes),len(self.nodes))))
    #fillQdiag() enforces (by modifying the diagonal of Q): sum of each row is zero
    def fillQdiag(self):
        numpy.fill_diagonal(self.Q,0.)
        Qdiag = -self.Q.sum(axis=1)
        numpy.fill_diagonal(self.Q,Qdiag)
    #addBiEdge() modifies parameters of a bidirectional transition and calls fillQdiag()
    def addBiEdge(self,first,second,q12,q21):
        self.Q[first, second] = q12
        self.Q[second, first] = q21
        self.fillQdiag()
    #addEdge() modifies parameters of a monodirectional transition and calls fillQdiag()
    def addEdge(self,first,second,q12):
        self.Q[first, second] = q12
        self.fillQdiag()
        
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