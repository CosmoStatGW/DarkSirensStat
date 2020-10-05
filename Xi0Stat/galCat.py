#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:32:31 2020

@author: Michi
"""

####
# This module contains a abstract classes to handle a galaxy catalogue
####

import pandas as pd
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from copy import deepcopy

from Xi0Stat.globals import *


class GalCat(ABC):
    
    def __init__(self, foldername, completeness, **kwargs):
        print('Initializing GalCat...')
        
        self._path = os.path.join(dirname, 'data', foldername)
        
        self._nside = 128
        
        self.data = pd.DataFrame()
        
        self.load()
        
        self._completeness = deepcopy(completeness)
        self._completeness.compute(self.data)
        
    
    @abstractmethod
    def load(self):
        pass
    
    def completeness(self, Omega, z):
        return self._completeness.get(Omega, z)
    
        
    def completeness_maps(self):
    
        zs = np.linspace(0.00, 0.5, 40)
        nPix = hp.nside2npix(512)
        px = np.arange(nPix)
        
        finemaps = self.completeness(hp.pix2ang(512, px), zs)
        
        for z in zs:
            
            finemap = self.completeness(hp.pix2ang(512, px), z)
                                        
            hp.mollview(finemap, notext=True)
            
            
            plt.savefig('z=' + str(z) + '.png')

    
class GalCompleted(object):
    
    def __init__(self, **kwargs):
        print('Initializing GalCompleted...')
        self._galcats = []
    
    def add_cat(self, cat):
        self._galcats.append(cat)
        
    def total_completeness(self, Omega, z):
        return sum(list(map(lambda c: c.completeness(Omega, z), self._galcats)))
    
    def confidence(compl):
        # multiplicative completion:
        #return 1
        # homogeneous completion
        #return 0
        # general interplation
        confpower = 0.05
        return np.exp(confpower*(1-1/compl))
        
    def get_inhom_contained(self, Omega, z):
        pass
    
    def eval_inhom(self, Omega, z):
        pass
        
    def eval_hom(self, Omega, z):
        pass
