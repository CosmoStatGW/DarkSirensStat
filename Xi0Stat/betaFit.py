#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:31:27 2020

@author: Michi
"""

from Xi0Stat.beta import Beta
from Xi0Stat.globals import *
from scipy.integrate import quad


class BetaFit(Beta):
    
    
    def __init__(self, **kwargs):
        Beta.__init__(self, **kwargs)
        
        
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
       
        
    
    
    def _get_beta(self, H0, Xi0, n=1.91, zR=zRglob, **kwargs):
        #print('zR get beta: %s' %zR)
        
        cosmo=FlatLambdaCDM(H0=H0GLOB, Om0=Om0GLOB)
        norm = quad(lambda x: cosmo.differential_comoving_volume(x).value, 0, zR )[0]
        
        num = quad(lambda x: BB(dLGW(x, H0=H0, Xi0=Xi0, n=n), **kwargs)*cosmo.differential_comoving_volume(x).value, 0, zR )[0]
        
        
        return num/norm
      
    
def BB(dL, gamma=gammaGlob, d0=d0GlobO2):
    #print('gamma BB: %s' %gamma)
    #print('d0 BB: %s' %d0)
    a0, a1 = get_as(gamma)
    Bfit = np.exp(-a0*(dL-d0)/d0-a1*((dL-d0)/d0)**2)
        
    return np.where(dL<=d0, 1, Bfit)


def get_as(gamma):
    a0= (5.21+9.55*gamma+3.47*gamma**2)*1e-02
    a1=(7.37-0.72*gamma)*1e-02
    return a0, a1
