#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:32:31 2020

@author: Michi
"""

####
# This module contains a abstract classes to handle a galaxy catalogue
####


from abc import ABC, abstractmethod
from copy import deepcopy

class GalCat(ABC):
    
    def __init__(self, completeness, **kwargs):
        print('Initializing GalCat...')
        
        self.load()
        
        self._completeness = deepcopy(completeness)
        self._completeness.compute()
        
    
    @abstractmethod
    def load(self):
        pass
    
    def completeness(self, Omega, z):
        return self._completeness.get(Omega, z)
    
    
    
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
        
    
    def get_inhom(self, Omega, z):
        pass
        
    def get_hom(self, Omega, z):
        pass
