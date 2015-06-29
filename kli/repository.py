class RepoField(object):
    def __init__(self):
        self.F = {}

    def getOrMake(self, assumed, true):
        try:
            L = self.F[(assumed, true)]
        except KeyError:
            L = []
            self.F[(assumed, true)] = L
        return L

    def trim(self, true, mReps):
        for key in self.F:
            if key[1] is true:
                del self.F[key][mReps:]


class Repository(object):
    def __init__(self):
        self.likes = RepoField()
        self.likeInfo = RepoField()

    def trim(self, true, mReps):
        self.likes.trim(true, mReps)
        self.likeInfo.trim(true, mReps)

class DataOnly(object):
    def __init__(self,data=None):
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

    def append(self,new):
        self.data.append(new)
        self.mReps = len(self.data)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return repr(self.data)


Repo = Repository()
