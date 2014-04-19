import numpy
import math
import copy
import pint

u = pint.UnitRegistry()

def v(x):
    try:
        return x.evaluate()
    except:
        return x
        
def setUnit(x,units):
    return x._magnitude * getattr(u,units)

class Parameter(object):
    def __init__(self,name,value=1.,units='dimensionless',default=1.,log=False):
        self.name = name
        # must define values so that setting routines work
        self.useLog = False
        self.value = 1. * u.dimensionless
        self.default = 1. * u.dimensionless
        if log:
            self.bounds = numpy.matrix([0, numpy.inf]) *u.dimensionless
            self.setLog()
        else:
            self.bounds = numpy.matrix([-numpy.inf, numpy.inf]) *u.dimensionless
        self.assign(value,units)
        self.setDefault(default)
        self.integrity()
    def __float__(self):
        return float(self.value._magnitude)
    def rename(self,name):
        self.name = name
        self.integrity()
    def setUnits(self,units):
        self.value = setUnit(self.value,units)
        self.default = setUnit(self.default,units)
        self.bounds = setUnit(self.bounds,units)
    def setLog(self):
        self.useLog = True
        if self.Lower() <0.:
            print("Warning: lower limit of range is negative; setting to zero")
            self.setLower(0.)
        self.integrity()
    def setLinear(self):
        self.useLog = False
        self.integrity()
    def evaluate(self):
        return self.value
    def assign(self,value,units=None):
        if not units==None:
            self.setUnits(units)
        self.value._magnitude = value
        self.checkValue()  # a weak version of integrity()
    def assignLog(self,logValue,units=None):
        if not units==None:
            self.setUnits(units)
        if self.useLog:
            self.value._magnitude = math.exp(logValue)
        else:
            self.value._magnitude = logValue
        self.checkValue() # a weak version of integrity()
    def setDefault(self, default):
        self.default._magnitude = default
        self.integrity()
    def setUpper(self, upper):
        self.bounds._magnitude[0,1] = upper
        self.integrity()
    def setLower(self,lower):
        self.bounds._magnitude[0,0] = lower
        self.integrity()
    def constrained(self):
        # returns True if there are constraints 
        # that is, if limits are different from [0,inf] for log, or [-inf,inf] for linear
        self.integrity()
        isR = not (numpy.isfinite(self.Lower()) or numpy.isfinite(self.Upper()))
        isRplus = (self.Lower() == 0) and (not numpy.isfinite(self.Upper()))
        if self.useLog:
            return not isRplus
        else:
            return not isR
    def setPositive(self):
        self.setLower(0.)
        self.setUpper(numpy.inf)
        self.integrity()
    def setReal(self):
        self.setLower(-numpy.inf)
        self.setUpper(numpy.inf)
        self.integrity()
    def __repr__(self):
        if (self.useLog):
            LL = '   Log scaling'
        else:
            LL = '   Linear scaling'
        if self.constrained():
            C = ' (constrained)'
        else:
            C = ' (unconstrained)'
        s = "Parameter: "+self.name+" = "+str(self.value)+'\n'
        s += LL + C + ' in '+str(self.bounds)
        s += '\n   Defaults to: '+str(self.default)
        return s
    def __str__(self):
        return self.name +" = "+str(self.value)
    def __add__(self, x):
        try:
            return self.evaluate() + x.evaluate()
        except:
            return self.evaluate() + x
    def __sub__(self, x):
        try:
            return self.evaluate() - x.evaluate()
        except:
            return self.evaluate() - x
    def __mul__(self, x):
        try:
            return self.evaluate() * x.evaluate()
        except:
            return self.evaluate() * x
    def __div__(self, x):
        try:
            return self.evaluate() / x.evaluate()
        except:
            return self.evaluate() / x
    def __pow__(self,x):
        try:
            return self.evaluate() ** x.evaluate()
        except:
            return self.evaluate() ** x
    def __radd__(self, x):
        try:
            return x.evaluate() + self.evaluate()
        except:
            return x + self.evaluate()
    def __rsub__(self, x):
        try:
            return x.evaluate() - self.evaluate()
        except:
            return x - self.evaluate()
    def __rmul__(self, x):
        try:
            return x.evaluate() * self.evaluate()
        except:
            return x * self.evaluate()
    def __rdiv__(self, x):
        try:
            return x.evaluate() / self.evaluate()
        except:
            return x / self.evaluate()
    def __rpow__(self, x):
        try:
            return x.evaluate() ** self.evaluate()
        except:
            return x ** self.evaluate()
    def getParameters(self):
        return {self.name:self}
    def getExpressions(self):
        return {}
    def Upper(self):
        return self.bounds._magnitude[0,1]
    def Lower(self):
        return self.bounds._magnitude[0,0]
    def Default(self):
        return self.default._magnitude
    def Value(self):
        return self.value._magnitude
    def checkValue(self):   # a weak version of integrity()
        assert(self.Lower() <= self.Value())
        assert(self.Value() <= self.Upper())
    def integrity(self):
        assert(isinstance(self.name,basestring))
        assert(self.Lower() <= self.Default())
        assert(self.Default() <= self.Upper())
        self.checkValue()
        if self.useLog:
            assert(self.Lower() >=0.)
            assert(self.Value() > 0.)
            assert(self.Default() > 0.)
            assert(self.Upper() > 0.)

class Space(object):
    def __init__(self,items):
        self.pDict = {}
        self.eDict = {}
        for i in items:
            self.pDict.update(i.getParameters())
            self.eDict.update(i.getExpressions())
        self.integrity()
    def __repr__(self):
        if not any(self.pDict):
            s = 'Empty Space'
        else:
            s = 'Space'
        for value in self.pDict.itervalues():
            s += '\n '+repr(value)
        for value in self.eDict.itervalues():
            s += '\n '+repr(Value)
        return s
    def __str__(self):
        if len(self.pDict)==0 and len(self.eDict) == 0:
            s = ' Empty Expression/Parameter Space'
        else:
            s = ''
        if len(self.eDict) > 0:
            s += ' Expressions:'
            for key, value in self.eDict.iteritems():
                s += "\n  "+key+" = "+str(value)+" = "+str(value.lastV)
        if len(self.pDict) > 0 and len(self.eDict) > 0:
            s += '\n'
        if len(self.pDict) > 0:
            s += ' Parameters:'
            for key, value in self.pDict.iteritems():
                s += "\n  "+str(value)
        return s
    def append(self,x):
        keys = pNameSet(self)
        newkeys = pNameSet(x)
        doubles = set.intersection(keys,newkeys)
        for d in doubles:
            if isinstance(x,Space):
                # parameter with same name: must have same identity
                assert(self.pDict[d] is x.pDict[d])
            else:
                # parameter with same name: must have same identity
                assert(self.pDict[d] is x)
        keys = eNameSet(self)
        newkeys = eNameSet(x)
        doubles = set.intersection(keys,newkeys)
        for d in doubles:
            if isinstance(x,Space):
                # parameter with same name: must have same identity
                assert(self.eDict[d] is x.eDict[d])
            else:
                # parameter with same name: must have same identity
                assert(self.eDict[d] is x)
        if isinstance(x,Space):
            self.pDict.update(x.pDict)
            self.eDict.update(x.eDict)
        else:
            self.pDict[x.name] = x
        self.integrity()
    def integrity(self):
        #make sure parameters have the right names 
        #   (in dictionary and name attribute of parameter)
        #also ensures they have different names
        for key, value in self.pDict.iteritems():
            assert(isinstance(value,Parameter))
            assert(key==value.name)
        for key, value in self.eDict.iteritems():
            assert(isinstance(value,Expression))
            assert(key==value.name)

def pNameSet(x):
    if isinstance(x,Space):
        return set(x.pDict.iterkeys())
    if isinstance(x,Parameter):
        return set(x.name)
    else:
        return set()
        
def eNameSet(x):
    if isinstance(x,Space):
        return set(x.eDict.iterkeys())
    if isinstance(x,Expression):
        return set(x.name)
    else:
        return set()
        
def emptySpace():
    return Space([])
        
class Expression(object):
    def __init__(self,name,expr,items):
        self.name = name
        self.expr = expr
        self.PS = Space(items)
        # the following command checks integrity & defines 
        #     self.value (frozen numeric value of of expression), and
        #     self.frozen (frozen params that created the value)
        self.thaw()  
    def __float__(self):
        if not self.frozen:
            self.evaluate()
        return float(self.lastV._magnitude)
    def __repr__(self):
        if not self.frozen:
            self.evaluate()
        s = 'Expression: '
        s += self.expr+" = "+str(self.lastV)+', where'
        if len(self.lastE) > 0:
            s += '\n Nested Expressions:'
        for key,value in self.lastE.iteritems():
            s += "\n   "+key+" = "+str(value)+" = "+str(value.lastV)
        if len(self.lastP) > 0:
            s += '\n Parameters:'
        for key,value in self.lastP.iteritems():
            s += "\n   "+key+" = "+str(value)
        return s
    def __str__(self):
        return self.expr
    def reexpress(self,E=None,P=None):
        if not E is None:
            self.expr = E
        if not P is None:
            self.PS = P
        self.integrity()
    def evaluate(self):
        methods = {"exp":__import__('math').exp,
                            "log":__import__('math').log,
                            "sin":__import__('math').sin,
                            "cos":__import__('math').cos,
                            "tan":__import__('math').tan,
                            "log10":__import__('math').log10,
                            "pi":__import__('math').pi,
                            "e":__import__('math').e,
                            "u":__import__('parameter').u,
                            "v":__import__('parameter').v}
        self.lastP = {}
        self.lastE = {}
        for key, v in self.PS.pDict.iteritems():
            self.lastP[key] = v
        for key,v in self.PS.eDict.iteritems():
            self.lastE[key] = v
        methods.update(self.lastP)
        methods.update(self.lastE)
        self.lastV = eval(self.expr,methods)
        return(self.lastV)
    def freeze(self):
        self.evaluate()
        self.frozen = True
    def thaw(self):
        self.evaluate()
        self.frozen = False
    def __add__(self, x):
        try:
            return self.evaluate() + x.evaluate()
        except:
            return self.evaluate() + x
    def __sub__(self, x):
        try:
            return self.evaluate() - x.evaluate()
        except:
            return self.evaluate() - x
    def __mul__(self, x):
        try:
            return self.evaluate() * x.evaluate()
        except:
            return self.evaluate() * x
    def __div__(self, x):
        try:
            return self.evaluate() / x.evaluate()
        except:
            return self.evaluate() / x
    def __pow__(self,x):
        try:
            return self.evaluate() ** x.evaluate()
        except:
            return self.evaluate() ** x
    def __radd__(self, x):
        try:
            return x.evaluate() + self.evaluate()
        except:
            return x + self.evaluate()
    def __rsub__(self, x):
        try:
            return x.evaluate() - self.evaluate()
        except:
            return x - self.evaluate()
    def __rmul__(self, x):
        try:
            return x.evaluate() * self.evaluate()
        except:
            return x * self.evaluate()
    def __rdiv__(self, x):
        try:
            return x.evaluate() / self.evaluate()
        except:
            return x / self.evaluate()
    def __rpow__(self, x):
        try:
            return x.evaluate() ** self.evaluate()
        except:
            return x ** self.evaluate()
    def getParameters(self):
        new = {}
        for v in self.PS.pDict.itervalues():
            new.update(v.getParameters())
        for v in self.PS.eDict.itervalues():
            new.update(v.getParameters())
        return new
    def getExpressions(self):
        new = {self.name:self}
        for v in self.PS.eDict.itervalues():
            new.update(v.getExpressions())
        return new

def getSpace(x):
    try:  # if object has a attribute names PS, return PS
        return x.PS
    except:
        pass
    if isinstance(x,Parameter):  # if it's a Parameter return singleton: Space(Param)
        return Space([x])
    if isinstance(x,Space):  # if it's a Space return the same object
        return x
    else:  # if all else fails return the empty space.
        return emptySpace()
        