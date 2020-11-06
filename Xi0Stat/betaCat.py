#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:35:42 2020

@author: Michi
"""

####
# This module contains a class to compute beta from the galaxy catalogue prior
####

from globals import *
from beta import Beta


class BetaCat(Beta):
    
    def __init__(self, gals, galRedshiftErrors,  zR, EventSelector, **kwargs):
        Beta.__init__(self, **kwargs)
        self.EventSelector=EventSelector
        self.gals=gals
        self.zR=zR
        self._galRedshiftErrors=galRedshiftErrors
        self._nside=1024
        
        
    def get_beta(self, H0s, Xi0s, n=nGlob, **kwargs):
        '''
        Computes beta 
        from eq. 2.134
        '''
        #print('-- %s' %GWevent.event_name)
        H0s = np.atleast_1d(H0s)
        Xi0s = np.atleast_1d(Xi0s)
        
        #self.gals.select_area(GWevent.selected_pixels, GWevent.nside)
        #self.gals.set_z_range_for_selection( *GWevent.get_z_lims(), return_count=False)
        self.gals.select_completeness(self.EventSelector)
        
        beta_cat = np.ones((H0s.size, Xi0s.size))
        beta_hom = np.ones((H0s.size, Xi0s.size))
        
        for i in np.arange(H0s.size):
            
            for j in np.arange(Xi0s.size):
                             
                beta_cat[i,j] = self._get_beta_cat( H0=H0s[i], Xi0=Xi0s[j], n=n, **kwargs)
                beta_hom[i,j] = self._get_beta_hom( H0=H0s[i], Xi0=Xi0s[j], n=n, **kwargs)
                                
        return np.squeeze(beta_cat+beta_hom) 
    
    
    
    def _get_beta_cat(self, H0, Xi0, n):
        
        if self._galRedshiftErrors:
            rGrid = np.linspace(0, self.zR, 500) #self._get_rGrid(GWevent.event_name, nsigma=GWevent.std_number, minPoints=20)

            zGrid = z_from_dLGW_fast(rGrid, H0=H0, Xi0=Xi0, n=n)
            
            _, weights = self.gals.get_inhom_contained(zGrid, self._nside )
            
            B = BB(rGrid)
            #skymap = self.selectedGWevents[eventName].likelihood_px(rGrid[np.newaxis, :], pixels[:, np.newaxis])
         
            #LL = np.sum(B*weights)
        else:
            _, zs, weights =  self.gals.get_inhom(self._nside)
            
            rs = dLGW(zs, H0=H0, Xi0=Xi0, n=n)
            
            B = BB(rs)#my_skymap = self.selectedGWevents[eventName].likelihood_px(rs, pixels)
            
        LL = np.sum(B*weights)
        return LL    
            
            
             
        
    def _get_beta_hom(self, H0, Xi0, n):
        
        return 0