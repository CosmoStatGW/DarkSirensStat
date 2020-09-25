#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:32:31 2020

@author: Michi
"""

####
# This module contains a abstract classes to handle a galaxy catalogue
####



class GalCat(object):
    
    def __init__(self, completeness, **kwargs):
        print('Initializing GalCat...')
        
        # call abstract load function
        
        # copy completness object
        # and compute completeness
        pass
    
    
    
    
    
class GalCompleted(object):
    
    def __init__(self, **kwargs):
        print('Initializing GalCompleted...')
        pass
    
    def add_cat(self, cat):
        pass
