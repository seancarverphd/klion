from unittest import TestCase
import kli.simple
import kli.repetitions

__author__ = 'sean'


class TestFlatSimple(TestCase):
  def setUp(self):
    self.S20 = kli.simple.Simple(n=20, p=.5)
    self.FS20 = self.S20.flatten(seed=17)
    self.R20 = kli.repetitions.Repetitions(self.FS20,9)
    self.R20.sim(100)
    self.FS20.sim()
    self.S21 = kli.simple.Simple(n=21, p=.5)
    self.FS21 = self.S21.flatten(seed=28)
    self.R21 = kli.repetitions.Repetitions(self.FS21,9)
    self.R21.sim(100)
    self.FS21.sim()

  def test_exact(self):
    B2_9 = kli.simple.Simple(n=2, p=.9)
    B2_8 = kli.simple.Simple(n=2, p=.8)
    EB2_9 = B2_9.exact()
    EB2_8 = B2_8.exact()
    self.assertEquals(EB2_9.KL(EB2_8), 0.073380028069501169)

  def test_Simple_OneRep_Simulate(self):
    self.assertEquals(self.FS21.data[81], 11)
    self.assertEquals(self.FS21.data[17], 8)

  def test_Simple_OneRep_Likelihood(self):
    self.assertEquals(self.FS21.like(), -2038.9092831205473)

  def test_Simple__OneRep_KL(self):
    self.assertEquals(self.FS21.KL(self.FS20), 0.027827403071797166)

  def test_Simple_Repetitions_Simulate(self):
    data21 = self.R21.get_data()
    data20 = self.R20.get_data()
    self.assertEquals(data21[81][2], 15)
    self.assertEquals(data20[17][2], 9)

  def test_Simple_Repetitions_Likelihood(self):
    self.assertEquals(self.R21.like(), -2038.9092831205546)

  def test_Simple__Repetitions_KL(self):
    self.assertEquals(self.R21.KL(self.R20), 0.25044662764617148)
