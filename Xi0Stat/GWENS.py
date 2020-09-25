#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:33:30 2020

@author: Michi
"""

####
# This module contains a class to handle the GLADE catalogue
####

from Xi0Stat.galCat import GalCat


class GWENS(GalCat):
    
    def __init__(self, compl, **kwargs):
        print('Initializing GWENS...')
        GalCat.__init__(self, compl, **kwargs)
        pass
