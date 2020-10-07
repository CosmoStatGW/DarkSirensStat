#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:28:47 2020

@author: Michi
"""

import unittest

# not needed after all
# import os, sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Xi0Stat.GLADE import GLADE
from Xi0Stat.GWENS import GWENS
from Xi0Stat.completeness import *
from Xi0Stat.galCat import GalCompleted

import numpy as np

class TestGal(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        
        compl = SkipCompleteness()
        cls.glade = GLADE('MINIGLADE', compl, band='K', Lcut=0.2, colnames_final = ['GWGC_name', 'z', 'K_Lum', 'w'])
        #GWENS('GWENS', compl, [22])
        cls.gwens   = GWENS('GWENS', compl)
        
    @classmethod
    def tearDownClass(cls):
        pass
        
    def setUp(self):
        self.gals = GalCompleted()

    def test_add_cat(self):
        self.gals.add_cat(TestGal.glade)
        self.gals.add_cat(TestGal.gwens)
        
    def test_total_completeness(self):
        self.gals.add_cat(TestGal.glade)
        
        # Checks that completenss at \Omega=(0, 0), z=0 is 1
        self.assertTrue(self.gals.total_completeness([0,0], 0) == 1)
        
        self.gals.add_cat(TestGal.gwens)
        
        # Checks that completenss at \Omega=(0, 0), z=0 is 2 when summing GLADE and GWENS
        self.assertTrue(self.gals.total_completeness([0,0], 0) == 2)
        
        # check corrected redshift of NGC4993
        NGC4993 = TestGal.glade.data[TestGal.glade.data['GWGC_name']=='NGC4993']
        print('TestGal GLADE check: NGC4993')
        print(NGC4993)
        self.assertTrue( np.round(NGC4993.z.values[0], 6) == 0.011026 )
        
if __name__ == '__main__':
    unittest.main()
