#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:35:42 2020

@author: Michi
"""

####
# This module contains a class to compute beta from the galaxy catalogue prior
####


from beta import Beta

class BetaCat(Beta):
    
    def __init__(self, **kwargs):
        print('Initializing BetaCat...')
        Beta.__ini__(**kwargs)
        pass