from unittest import TestCase
import kli.toy
__author__ = 'sean'


class TestSaveSeedRNG(TestCase):
    def setUp(self):
        self.R00 = kli.toy.SaveSeedRNG()
        self.R00.setSeedAndOffset(0,0)
        self.R10 = kli.toy.SaveSeedRNG()
        self.R10.setSeedAndOffset(1,0)
        self.R01 = kli.toy.SaveSeedRNG()
        self.R01.setSeedAndOffset(0,1)
        self.r00 = self.R00.normalvariate(0,1)
        self.r01 = self.R01.normalvariate(0,1)
        self.r10 = self.R10.normalvariate(0,1)

    def test_setOffset(self):
        self.assertEquals(self.r01,self.r10)

    def test_reset(self):
        R = kli.toy.SaveSeedRNG()
        a = R.normalvariate(0,1)
        b = R.normalvariate(0,1)
        R.reset()
        c = R.normalvariate(0,1)
        self.assertEquals(a,c)
        self.assertNotEquals(a,b)
