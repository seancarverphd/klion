import channel
import numpy

class Patch(object):
    def __init__(self, channels):
        self.channels = channels
        self.assertOneChannel()
    def assertOneChannel(self)
        assert(len(channels) == 1)
        assert(channels[0][0] == 1)
        ch = channels[0][1]
        assert(isinstance(ch,channel.Channel))
