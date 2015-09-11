from unittest import TestCase
import kli.toy
__author__ = 'sean'


class TestSaveSeedRNG(TestCase):
    def setUp(self):
        self.R00 = kli.toy.SaveStateRNG()
        self.R00.reseed(0)
        self.R10 = kli.toy.SaveStateRNG()
        self.R10.reseed(1)
        self.r00 = self.R00.normal(0,1)
        self.r10 = self.R10.normal(0,1)

    def test_reset(self):
        R = kli.toy.SaveStateRNG()
        a = R.normal(0,1)
        b = R.normal(0,1)
        R.reset()
        c = R.normal(0,1)
        self.assertEquals(a,c)
        self.assertNotEquals(a,b)
