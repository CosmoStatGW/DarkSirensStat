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


class TestGal(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        compl = SkipCompleteness()
        cls.glade = GLADE(compl)
        cls.gwens   = GWENS(compl)
        
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
    
        self.assertTrue(self.gals.total_completeness([0,0], 0) == 1)
        
        self.gals.add_cat(TestGal.gwens)
     
        self.assertTrue(self.gals.total_completeness([0,0], 0) == 2)
        
if __name__ == '__main__':
    unittest.main()
