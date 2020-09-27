#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:33:53 2020

@author: Michi
"""

####
# This module contains objects to compute the completeness of a galaxy catalogue
####

from abc import ABC, abstractmethod

class Completeness(ABC):
    
    def __init__(self, **kwargs):
        print('Initializing Completeness...')
        pass
        
    def compute(self):
        print('Computing completeness')
        self.compute_implementation()
        self._computed = True
    
    @abstractmethod
    def compute_implementation(self):
        pass
    
    def get(self, Omega, z):
        assert(self._computed)
        return self.get_implementation(Omega, z)
        
    @abstractmethod
    def get_implementation(self, Omega, z):
        pass
    
    
    
class SkipCompleteness(Completeness):
    
    def __init__(self, **kwargs):
        print('Initializing SkipCompleteness...')
        Completeness.__init__(self, **kwargs)
        pass
        
    def compute_implementation(self):
        pass
    
    def get_implementation(self, Omega, z):
        return 1
        

    
class LocalCompleteness(Completeness):
    
    def __init__(self, **kwargs):
        print('Initializing LocalCompleteness...')
        Completeness.__init__(**kwargs)
        pass
        
    def compute_implementation(self):
        pass
    
    def get_implementation(self):
        return 1
