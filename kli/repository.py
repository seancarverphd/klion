class TableOfModels(object):
    def __init__(self):
        self.table = {}

    def getOrMakeEntry(self, model):
        try:
            entry = self.table[model]
        except KeyError:
            entry = []
            self.table[model] = entry
        return entry

    def trim(self, mReps=0):
        for model in self.table:
            del self.table[model][mReps:]


class DataOnly(object):
    def __init__(self, data=None):
        if data is None:
            self.data = []
        else:
            self.data = data
        self.mReps = len(self.data)
        self.base = self

    def sim(self, mReps=None):
        if mReps is None:
            self.mReps = len(self.data)
        elif mReps > len(self.data):
            assert False
        else:
            self.mReps = mReps

    def append(self, new):
        self.data.append(new)
        self.mReps = len(self.data)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return repr(self.data)
