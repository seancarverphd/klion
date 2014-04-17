import numpy
import math
import copy
import pint

u = pint.UnitRegistry()

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
        return str(self.value)
    def __float__(self):
        return float(self.value)
    def __add__(self, x):
        return self.value + x.value
    def __sub__(self, x):
        return self.value - x.value
    def __mul__(self, x):
        return self.value * x.value
    def __div__(self, x):
        return self.value / x.value
    def __pow__(self,x):
        return self.value ** x.value
    def __radd__(self, x):
        return x.value + self.value
    def __rsub__(self, x):
        return x.value - self.value
    def __rmul__(self, x):
        return x.value * self.value
    def __rdiv__(self, x):
        return x.value / self.value
    def __rpow__(self, x):
        return x.value ** self.value
    # The next two functions work, but I haven't decided if they are a good idea
    #~ def __rxor__(self, x):  # So you can type B^A for B**A
        #~ return float(x)**float(self)
    #~ def __xor__(self,x):  # So you can type A^B for A**B
        #~ return float(self)**float(x)
    def Upper(self):
        return self.bounds._magnitude[0,1]
    def Lower(self):
        return self.bounds._magnitude[0,0]
    def Default(self):
        return self.default._magnitude
    def Value(self):
        return self.default._magnitude
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
    def __init__(self,x):
        try:
            self.pDict = {}
            for p in x:
                self.pDict[p.name] = p
            print self.pDict
        except:  # assume it is already a dictionary (fix: make sure of this)
            self.pDict = x
        self.integrity()
    def __repr__(self):
        if not any(self.pDict):
            s = 'Empty Parameter Space'
        else:
            s = 'Parameter Space'
        for value in self.pDict.itervalues():
            s += '\n '+repr(value)
        return s
    def __str__(self):
        s = 'Parameter Space'
        for value in self.pDict.itervalues():
            whole = repr(value)
            first = whole.split('\n',1)[0]
            s += '\n '+str(first)
        return s
    def append(self,x):
        keys = nameSet(self)
        newkeys = nameSet(x)
        doubles = set.intersection(keys,newkeys)
        for d in doubles:
            if isinstance(x,Space):
                # parameter with same name: must have same identity
                assert(self.pDict[d] is x.pDict[d])
            else:
                # parameter with same name: must have same identity
                assert(self.pDict[d] is x)
        if isinstance(x,Space):
            self.pDict.update(x.pDict)
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

def nameSet(x):
    if isinstance(x,Space):
        return set(x.pDict.iterkeys())
    try:
        if isinstance(x.PS,Space):
            return namesOfParmas(x.PS)
    except:
        pass
    if isinstance(x,Parameter):
        return set(x.name)
    else:
        return set()
        
def emptySpace():
    return Space({})
        
class Expression(object):
    def __init__(self,expr,PS):
        self.expr = expr
        self.PS = PS
        # the following command checks integrity & defines 
        #     self.value (frozen numeric value of of expression), and
        #     self.frozen (frozen params that created the value)
        self.freeze()  
    def __float__(self):
        if not self.frozen:
            self.evaluate()
        return self.lastV
    def __repr__(self):
        s = 'Expression: '
        s += self.expr+" = "+str(float(self))+', where'
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
                            "e":__import__('math').e}
        self.lastP = {}
        for key, v in self.PS.pDict.iteritems():
            self.lastP[key] = float(v)
        methods.update(self.lastP)
        self.lastV = float(eval(self.expr,methods))
    def freeze(self):
        self.evaluate()
        self.frozen = True
    def thaw(self):
        self.frozen = False

def getSpace(x):
    try:  # if object has a attribute names PS, return PS
        return x.PS
    except:
        pass
    if isinstance(x,Parameter):  # if it's a Parameter return singleton: Space(Param)
        return Space({x.name:x})
    if isinstance(x,Space):  # if it's a Space return the same object
        return x
    else:  # if all else fails return the empty space.
        return emptySpace()
        