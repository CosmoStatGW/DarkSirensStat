#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:31:27 2020

@author: Michi
"""

from beta import Beta
from globals import *
from scipy.integrate import quad


class BetaFit(Beta):
    
    
    def __init__(self,  zR, **kwargs):
        Beta.__init__(self, **kwargs)
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
       
        
    
    
    def _get_beta(self, H0, Xi0, n=1.91, **kwargs):
        #print('zR get beta: %s' %zR)
        
        cosmo=FlatLambdaCDM(H0=H0GLOB, Om0=Om0GLOB)
        norm = quad(lambda x: cosmo.differential_comoving_volume(x).value, 0, self.zR )[0]
        
        num = quad(lambda x: BB(dLGW(x, H0=H0, Xi0=Xi0, n=n), **kwargs)*cosmo.differential_comoving_volume(x).value, 0, self.zR )[0]
        
        
        return num/norm
      
    
