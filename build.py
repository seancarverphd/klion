import numpy

class Level(object):
    def __init__(self,name,conductance,noise):
        assert(isinstance(name,basestring))
        conductance = float(conductance)
        noise = float(noise)
        self.name = name
        self.conductance = conductance
        self.noise = noise

class Node(object):
    def __init__(self,name,level):
        assert(isinstance(name,basestring))
        assert(isinstance(level,Level))
        self.name = name
        self.level = level
        
class Channel(object):
    def __init__(self,nodes):
        for node in nodes:
            assert(isinstance(node,Node))
        self.nodes = nodes
        self.Q = numpy.zeros(shape=(len(self.nodes),len(self.nodes)))
        
    def addNode(self,new):
        self.nodes.append(new)
    
    #def addEdge(self,one,two,q12,q21):