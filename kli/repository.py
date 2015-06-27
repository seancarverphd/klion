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


class Repository(object):
    def __init__(self):
        self.likes = RepoField()
        self.likeInfo = RepoField()

class DataOnly(object):
    def __init__(self):
        self.data = []
        self.mReps = 0
        self.base = self
    def append(self,new):
        self.data.append(new)
        self.mReps = len(self.data)
    def __str__(self):
        return str(self.data)
    def __repr__(self):
        return repr(self.data)


Repo = Repository()
