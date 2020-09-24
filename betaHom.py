#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:36:12 2020

@author: Michi
"""

####
# This module contains a class to compute beta from the assumption of homogeneous galaxy distribution
####


from beta import Beta

class BetaHom(Beta):
    
    def __init__(self, **kwargs):
        print('Initializing BetaHom...')
        Beta.__ini__(**kwargs)
        pass