#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:35:09 2020

@author: Michi
"""

####
# This module contains a class to compute beta from a realistic detection model from MC integration
####

from beta import Beta

class BetaMC(Beta):
    
    def __init__(self, **kwargs):
        print('Initializing BetaMC...')
        Beta.__ini__(**kwargs)
        pass