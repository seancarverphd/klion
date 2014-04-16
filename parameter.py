import numpy
import math
import copy
import pint

u = pint.UnitRegistry()

def setUnit(x,units):
    return x._magnitude * getattr(u,units)

class Parameter(object):
    def __init__(self,name):
        self.name = name
        self.useLog = False
        # if (useLog): then the next 4 still on linear scale
        self.value = 1.*u.dimensionless
        self.default = 1.*u.dimensionless
        self.lower = -numpy.inf*u.dimensionless
        self.upper = numpy.inf*u.dimensionless
        self.integrity()
    def rename(self,name):
        self.name = name
        self.integrity()
    def setUnits(self,units):
        self.value = setUnit(self.value,units)
        self.default = setUnit(self.default,units)
        self.lower = setUnit(self.lower,units)
        self.upper = setUnit(self.upper,units)
    def setLog(self):
        self.useLog = True
        if self.lower.magnitude<0.:
            print("Warning: lower limit of range is negative; setting to zero")
            self.lower._magnitude = 0.
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
    def setDefault(default):
        set.default = default
        self.integrity()
    def setUpper(self, upper):
        self.upper = upper
        self.integrity()
    def setLower(self,lower):
        self.lower = lower
        self.integrity()
    def constrained(self):
        # returns True if there are constraints 
        # that is, if limits are different from [0,inf] for log, or [-inf,inf] for linear
        self.integrity()
        isR = not (numpy.isfinite(self.lower) or numpy.isfinite(self.upper))
        isRplus = (self.lower == 0) and (not numpy.isfinite(self.upper))
        if self.useLog:
            return not isRplus
        else:
            return not isR
    def setPositive(self):
        self.lower = 0
        self.upper = numpy.inf
        self.integrity()
    def setReal(self):
        self.lower = -numpy.inf
        self.upper = numpy.inf
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
        s += LL + C + ' in ['+str(self.lower)+','+str(self.upper) +']'
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
    def checkValue(self):   # a weak version of integrity()
        assert(self.lower <= self.value)
        assert(self.value <= self.upper)
    def integrity(self):
        assert(isinstance(self.name,basestring))
        assert(self.lower < self.upper)
        assert(self.lower <= self.default)
        assert(self.default <= self.upper)
        self.checkValue()
        if self.useLog:
            assert(self.lower.magnitude >=0.)
            assert(self.value.magnitude > 0.)
            assert(self.default.magnitude >0.)
            assert(self.upper.magnitude > 0.)
            
class Space(object):
    def __init__(self,pDict):
        self.pDict = pDict
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
        