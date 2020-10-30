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
from globals import *
from scipy.integrate import quad

class BetaHom(Beta):
    
    def __init__(self, dMax, zR, **kwargs):
        Beta.__init__(self, **kwargs)
        self.dMax=dMax
        self.zR=zR
        
    
    def get_beta(self, H0s, Xi0s, n=nGlob, **kwargs):
        '''
        Computes beta 
        from eq. 2.134
        '''
        
        H0s = np.atleast_1d(H0s)
        Xi0s = np.atleast_1d(Xi0s)
        
        
        beta = np.ones((H0s.size, Xi0s.size))
        
        for i in np.arange(H0s.size):
            
            for j in np.arange(Xi0s.size):
                             
                beta[i,j] = self._get_beta( H0=H0s[i], Xi0=Xi0s[j], n=n, **kwargs)
                                
        return np.squeeze(beta) 
        
    def _get_beta(self, H0, Xi0,  n=1.91, ):
        '''
        Homogeneous beta -  Eq. 2.81 
        '''
        #  dmax is the maximum value of the luminosity distance 
        # (averaged over the solid angle and over the θ′ parameters, 
        # so in particular over the source inclination) 
        # to which the given monochromatic population of source could be detected,
        # given the sensitivity of the detector network
        # zMax is the corespondig
        
        zMax = z_from_dLGW(self.dMax, H0,  Xi0, n=n)
        cosmo=FlatLambdaCDM(H0=H0, Om0=Om0GLOB)
                
        norm = quad(lambda x: cosmo.differential_comoving_volume(x).value, 0, self.zR )[0]
        num = quad(lambda x:  cosmo.differential_comoving_volume(x).value, 0, zMax )[0]
        return num/norm
        
        return beta
