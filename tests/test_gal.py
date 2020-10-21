#!/usr/bin/env python3


import unittest

# not needed after all
# import os, sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Xi0Stat.GLADE import GLADE
from Xi0Stat.GWENS import GWENS
from Xi0Stat.SYNTH import SYNTH
from Xi0Stat.completeness import *
from Xi0Stat.galCat import GalCompleted

import numpy as np


class TestGal(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        
        compl = SkipCompleteness()
        verbose = False
        cls.glade = GLADE('MINIGLADE', compl, False, verbose=verbose, band='K', Lcut=0.2, colnames_final=['RA','dec', 'theta', 'phi','z', 'GWGC_name', 'w', 'K_Lum'])
        #GWENS('GWENS', compl, [22])
        #cls.gwens   = GWENS('GWENS', compl, False)
    
    
    @classmethod
    def tearDownClass(cls):
        pass
    
    def setUp(self):
        self.gals = GalCompleted()
    
    def test_add_cat(self):
        self.gals.add_cat(TestGal.glade)
        self.gals.add_cat(TestGal.glade)
        self.assertTrue(len(self.gals._galcats)==2)
    
    
    def test_data_GLADE(self):
        #print('\ntest_data_GLADE....')
        # check corrected redshift of NGC4993
        NGC4993 = self.glade.data[self.glade.data['GWGC_name']=='NGC4993']
        #print('TestGal GLADE check: NGC4993')
        #print(NGC4993)
        self.assertTrue( np.round(NGC4993.z.values[0], 6) == 0.011026 )
        
    
    def test_weights_GWENS(self):
        #print('\ntest_weights_GWENS....')
        #gwens = self.gwens.data
        #print(gwens.head(2))
        #self.assertTrue( np.abs(np.sum(gwens.w) - len(gwens.data))/len(gwens.data) < 1e-6)
        #print('test_weights_GWENS done.\n')
        pass



class TestComplGeneral(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        compl = SkipCompleteness()
        verbose = False
        cls.glade = GLADE('MINIGLADE', compl, False, verbose=verbose, band='K', Lcut=0.2)
    
    
    @classmethod
    def tearDownClass(cls):
        pass
    
    def setUp(self):
        self.gals = GalCompleted()
        self.gals.add_cat(TestComplGeneral.glade)
    
    def test_completeness_1pt(self):
        self.assertTrue(TestComplGeneral.glade.completeness(0,0, 0) <= 1)

        
    def test_total_completeness(self):

        #self.gals.add_cat(TestGal.glade)
        
        # total SkipCompleteness for one catalog is 1
        self.assertTrue(np.abs(self.gals.total_completeness(0,0, 0) - 1) < 1e-6)
    
        self.gals.add_cat(TestComplGeneral.glade)

        # total SkipCompleteness for two catalogs is 2
        self.assertTrue( np.abs(self.gals.total_completeness(0,0, 0) - 2) < 1e-6)
        
        self.gals.add_cat(TestComplGeneral.glade, weight=10)
        
        # total SkipCompleteness for 3 catalogs with nontrivial weight
        self.assertTrue( np.abs(self.gals.total_completeness(0,0, 0) - 12) < 1e-6)

  

class TestSuperpixelCompleteness(unittest.TestCase):
      
    @classmethod
    def setUpClass(cls):
        verbose = True
        comovingDensityGoal = get_SchNorm(phistar=phiBstar07, Lstar=LBstar07, alpha=alphaB07, Lcut=0.6)
        compl = SuperpixelCompleteness(comovingDensityGoal=comovingDensityGoal, angularRes=4, zRes=10, interpolateOmega = False)
        complInterp = SuperpixelCompleteness(comovingDensityGoal=comovingDensityGoal, angularRes=4, zRes=10, interpolateOmega = True)

        cls.synth = SYNTH(compl, verbose=verbose, useDirac = False, zMax = 0.005, comovingNumberDensityGoal=0.1)
        cls.synthDirac = SYNTH(compl, verbose=verbose, useDirac = True, zMax = 0.005, comovingNumberDensityGoal=0.1)
        
        cls.synthInterp = SYNTH(complInterp, verbose=verbose, useDirac = False, zMax = 0.005, comovingNumberDensityGoal=0.1)
        cls.synthDiracInterp = SYNTH(complInterp, verbose=verbose, useDirac = True, zMax = 0.005, comovingNumberDensityGoal=0.1)
        
      
      
    @classmethod
    def tearDownClass(cls):
        pass
  
    def setUp(self):
        pass
  
    def test(self):
        pass


if __name__ == '__main__':
    unittest.main()
