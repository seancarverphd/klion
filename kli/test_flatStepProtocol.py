from unittest import TestCase
import kli.channel
import kli.parameter
import kli.patch
import kli.engine
import numpy as np
from kli import u

__author__ = 'sean'


class TestFlatStepProtocol(TestCase):
    def setUp(self):
        # This code sets up a canonical channel
        # EK is Hodgkin Huxley value take from http://icwww.epfl.ch/~gerstner/SPNM/node14.html
        self.EK = kli.parameter.Parameter("EK", -77, "mV", log=False)
        # I = gV
        # gmax_khh is from www.neuron.yale.edu, but is a density parameter inappropriate for a single channel; use g_open instead
        self.gmax_khh = kli.parameter.Parameter("gmax_khh", 0.02979, "microsiemens", log=True)
        # "The single-channel conductance of typical ion channels ranges from 0.1 to 100 pS (picosiemens)."  Bertil Hille (2008), Scholarpedia, 3(10):6051.
        # For now, g_open is used only for plotting
        self.g_open = kli.parameter.Parameter("g_open", 1., "picosiemens", log=True)
        # The following two parameters were made up (but they are not used at the moment):
        self.gstd_open = kli.parameter.Parameter("gstd_open", 0.1, "picosiemens", log=True)
        self.gstd_closed = kli.parameter.Parameter("gstd_closed", 0.01, "picosiemens", log=True)

        # The rest of these parameters come from www.neuron.yale.edu (khh channel) channel builder tutorial
        self.ta1 = kli.parameter.Parameter("ta1", 4.4, "ms", log=True)
        self.tk1 = kli.parameter.Parameter("tk1", -0.025, "1/mV", log=False)
        self.d1 = kli.parameter.Parameter("d1", 21., "mV", log=False)
        self.k1 = kli.parameter.Parameter("k1", 0.2, "1/mV", log=False)

        self.ta2 = kli.parameter.Parameter("ta2", 2.6, "ms", log=True)
        self.tk2 = kli.parameter.Parameter("tk2", -0.007, "1/mV", log=False)
        self.d2 = kli.parameter.Parameter("d2", 43, "mV", log=False)
        self.k2 = kli.parameter.Parameter("k2", 0.036, "1/mV", log=False)

        self.V0 = kli.parameter.Parameter("V0", -65., "mV", log=False)
        self.V1 = kli.parameter.Parameter("V1", 20., "mV", log=False)
        self.V2 = kli.parameter.Parameter("V2", -80., "mV", log=False)
        # The parameter VOLTAGE is set by voltage-clamp in patch.py
        self.VOLTAGE = kli.parameter.Parameter("VOLTAGE", -65., "mV", log=False)
        self.OFFSET = kli.parameter.Parameter("OFFSET", 65., "mV", log=False)
        self.VOLTAGE.remap(self.V0)

        self.vr = kli.parameter.Expression("vr", "VOLTAGE + OFFSET", [self.VOLTAGE, self.OFFSET])
        self.tau1 = kli.parameter.Expression("tau1", "ta1*exp(tk1*vr)", [self.ta1, self.tk1, self.vr])
        self.K1 = kli.parameter.Expression("K1", "exp((k2*(d2-vr))-(k1*(d1-vr)))", [self.k1, self.k2, self.d1, self.d2, self.vr])
        self.tau2 = kli.parameter.Expression("tau2", "ta2*exp(tk2*vr)", [self.ta2, self.tk2, self.vr])
        self.K2 = kli.parameter.Expression("K2", "exp(-(k2*(d2-vr)))", [self.k2, self.d2, self.vr])

        self.a1 = kli.parameter.Expression("a1", "K1/(tau1*(K1+1))", [self.K1, self.tau1])
        self.b1 = kli.parameter.Expression("b1", "1/(tau1*(K1+1))", [self.K1, self.tau1])
        self.a2 = kli.parameter.Expression("a2", "K2/(tau2*(K2+1))", [self.K2, self.tau2])
        self.b2 = kli.parameter.Expression("b2", "1/(tau2*(K2+1))", [self.K2, self.tau2])

        self.Open = kli.channel.Level("Open", mean=self.g_open, std=self.gstd_open)
        self.Closed = kli.channel.Level("Closed", mean=0. * u.picosiemens, std=self.gstd_closed)
        self.C1 = kli.channel.Node("C1", self.Closed)
        self.C2 = kli.channel.Node("C2", self.Closed)
        self.O = kli.channel.Node("O", self.Open)
        self.khh = kli.channel.Channel([self.C1, self.C2, self.O])
        self.khh.biEdge("C1", "C2", self.a1, self.b1)
        self.khh.edge("C2", "O", self.a2)
        self.khh.edge("O", "C2", self.b2)
        self.khhPatch = kli.patch.singleChannelPatch(self.khh, self.VOLTAGE)  # set to channel.VOLTAGE for previous bug
        self.SP = kli.patch.StepProtocol(self.khhPatch, [-65*u.mV, -20*u.mV], [np.inf, 10*u.ms])
        self.FS = self.SP.flatten(5)
        self.FS.sim(10)

    def test_patch_like(self):
        self.assertEquals(kli.patch.FS.like(),-168.873183577661)

    # The following test passed when patch was setting VOLTAGE
    # according to channel.VOLTAGE not self.VOLTAGE
    # def test_self_like(self):
    #      self.assertEquals(self.FS.like(),-0.517332103313697)

    def test_same_construction(self):
        # This test used to fail because patch was setting VOLTAGE
        # according to channel.VOLTAGE not self.VOLTAGE
        self.assertEquals(self.FS.nReps, 10)
        self.assertEqual(kli.patch.FS.like(), self.FS.like())

    # def test_old_patch_like(self):
    #     self.assertEquals(kli.patch.FS.like(),-26.748642946985434)
    #
    # RESULT HERE GOT MULTIPLIED BY 10:
    #
    # def test_old_self_like(self):
    #     self.assertEquals(self.FS.like(),-0.0517332103313697)
    #
    # def test_old_same_construction(self):
    #     self.assertEquals(self.FS.nReps, 10)
    #     self.assertEqual(kli.patch.FS.like(), self.FS.like())

    def test_q65(self):
        Q = self.khhPatch.makeQ(-65*u.mV)._magnitude
        self.assertEquals(Q[0, 0], -0.014969510799849182)
        self.assertEquals(Q[2, 0], 0.)
        self.assertEquals(Q[0, 2], 0.)
        self.assertEquals(Q[1, 1], -0.27975526187055599)
        self.assertEquals(Q[2, 2], -0.31716333921770673)
        self.assertEquals(Q[1, 0], 0.21230321647287806)