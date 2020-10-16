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
from Xi0Stat.globals import *
from scipy.integrate import quad

class BetaHom(Beta):
    
    def __init__(self, **kwargs):
        print('Initializing BetaHom...')
        Beta.__init__(**kwargs)
        
        
    def get_beta(self, H0, Xi0, dMax, zR=5, n=1.91, ):
        '''
        Homogeneous beta -  Eq. 2.81 
        '''
        #  dmax is the maximum value of the luminosity distance 
        # (averaged over the solid angle and over the θ′ parameters, 
        # so in particular over the source inclination) 
        # to which the given monochromatic population of source could be detected,
        # given the sensitivity of the detector network
        # zMax is the corespondig
        
        zMax = z_from_dLGW(dMax, H0,  Xi0, n=n)
        cosmo=FlatLambdaCDM(H0=H0, Om0=Om0GLOB)
                
        norm = quad(lambda x: cosmo.differential_comoving_volume(x).value, 0, zR )[0]
        num = quad(lambda x:  cosmo.differential_comoving_volume(x).value, 0, zMax )[0]
        return num/norm
        
        return beta