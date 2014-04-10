import numpy

# A Level is a defined (mean, std) that will be the same across certain Nodes (ie states)
class Level(object):
    def __init__(self,name,mean,std):
        self.name = name
        self.mean = mean
        self.std = std
        self.integrity()
    def __repr__(self):
        return 'Level %s: (mean %s, std %s)' % (self.name,repr(self.mean),repr(self.std))
    def integrity(self):
        assert(isinstance(self.name,basestring))
        self.mean = float(self.mean)
        self.std = float(self.std)
        assert(self.std >= 0)
        
# A Node is a state of the channel
class Node(object):
    def __init__(self,name,level):
        self.name = name
        self.level = level
        self.integrity()
    def __repr__(self):
        return 'Node %s: %s' % (self.name, repr(self.level))
    def integrity(self):
        assert(isinstance(self.name,basestring))
        assert(isinstance(self.level,Level))
        self.level.integrity()
        
# A Channel is a model of an ion channel
class Channel(object):
    def __init__(self,nodes):
        self.nodes = nodes
        self.disconnect()  #defines self.Q = zero-matrix
        self.integrity()
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
        q12 = float(q12)
        q21 = float(q21)
        assert(q12>=0.)
        assert(q21>=0.)
        self.Q[first, second] = q12
        self.Q[second, first] = q21
        self.fillQdiag()
    #addEdge() modifies parameters of a monodirectional transition and calls fillQdiag()
    def addEdge(self,first,second,q12):
        q12 = float(q12)
        assert(q12>=0.)
        self.Q[first, second] = q12
        self.fillQdiag()
    def integrity(self): # Checks that channel is well defined
        #Nodes
        for n in self.nodes:
            assert(isinstance(n,Node))
            n.integrity()
        #Edges
        numpy.fill_diagonal(self.Q,0.)
        assert(numpy.amin(self.Q)==0)
        self.fillQdiag()
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