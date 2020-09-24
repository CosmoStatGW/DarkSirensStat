#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:33:30 2020

@author: Michi
"""

####
# This module contains a class to handle the GLADE catalogue
####

from Xi0Stat.galCat import Gal


class GLADE(Gal):
    
    def __init__(self, **kwargs):
        print('Initializing GLADE...')
        Gal.__init__(self, **kwargs)
        pass