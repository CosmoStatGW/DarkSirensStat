#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:33:53 2020

@author: Michi
"""

####
# This module contains objects to compute the completeness of a galaxy catalogue
####


class Completeness(object):
    
    def __init__(self, **kwargs):
        print('Initializing Completeness...')
        pass
    
    
class SkipCompleteness(Completeness):
    
    def __init__(self, **kwargs):
        print('Initializing SkipCompleteness...')
        Completeness.__init__(self, **kwargs)
        pass

    
class LocalCompleteness(Completeness):
    
    def __init__(self, **kwargs):
        print('Initializing LocalCompleteness...')
        Completeness.__init__(**kwargs)
        pass
