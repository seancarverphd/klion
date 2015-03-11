from unittest import TestCase
import unittest
import kli.toy
import kli.parameter

__author__ = 'sean'


class TestFlatToyProtocol(TestCase):
    def setUp(self):
        self.q0 = kli.parameter.Parameter("q0", 0.5, "kHz", log=True)
        self.q1 = kli.parameter.Parameter("q1", 0.25, "kHz", log=True)
        self.q = kli.parameter.Parameter("q", 1. / 6., "kHz", log=True)
        self.T3 = kli.toy.toyProtocol([self.q0, self.q1])
        self.T2 = kli.toy.toyProtocol([self.q])
        self.F2 = self.T2.flatten(2)
        self.F3 = self.T3.flatten(3)
        self.F2.sim(10)
        self.F3.sim(10)

    def test_like2(self):
        self.assertEqual(self.F2.like(), -30.619490619218585)

    def test_like3(self):
        self.assertEqual(self.F3.like(), -27.500748311455272)


if __name__ == '__main__':
    unittest.main()