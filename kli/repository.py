import numpy

class Repository(object):
    def __init__(self):
        dataset = {}  # (basemodel,seed) : data
        likeset = {}