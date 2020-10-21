#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:34:25 2020

@author: Michi
"""

####
# This module contains an abstract class to compute beta
####
from abc import ABC, abstractmethod

class Beta(ABC):
    
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def get_beta(self):
        pass        
        