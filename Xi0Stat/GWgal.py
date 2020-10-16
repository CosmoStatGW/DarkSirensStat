#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:51:19 2020

@author: Michi
"""

####
# This module contains a class to handle GW-galaxy correlation and compute the likelihood
####
from Xi0Stat.globals import *


class GWgal(object):
    
    def __init__(self, GalCompleted, GWevents,):
        
        self.gals = GalCompleted
        self.GWevents = GWevents
        self.cred_pixels = {event: self.GWevents[event].get_credible_region_pixels(level=0.99) for event in self.GWevents}
        self.z_lims = {event: self.GWevents[event].get_zlims() for event in self.GWevents}
        
        
        # Note on the generalization. Eventually we should have a dictionary
        # {'GLADE': glade catalogue, 'DES': ....}
        # and a dictionary {'GW event': 'name of the catalogue to use'}
        # The function _inhom_lik will use the appropriate catalogue according to
        # this dictionary
        
        # Completeness needs a name or something to know if we use 
        # multiplicative, homogeneous or no completion
        
        print('\n --- GW events: ')
        for event in GWevents.keys():
            print(event)
            
    
    
    def get_inhom_lik(self, H0, Xi0, n=1.91):
        '''
        Computes likelihood with p_cat for all events
        Returns dictionary {event_name: L_cat }
        '''    
        
        return {event: self._inhom_lik(event, H0=H0, Xi0=Xi0, n=n) for event in self.GWevents} 
     
    
    
    def get_hom_lik(self, H0, Xi0, n=1.91):
        '''
        Computes likelihood with homogeneous part for all events.
        Returns dictionary {event_name: L_hom }
        '''
        return {event: self._hom_lik(event, H0=H0, Xi0=Xi0, n=n) for event in self.GWevents}        
            
    
    
    def _inhom_lik(self, event, H0, Xi0, n=1.91):
        '''
        Computes likelihood with p_cat for one event
        Output: np array of dim (N. galaxies in 99% credible region, 1)
        '''
        
        # Remebmer to set gals somewhere:
        #self.gals.set_z_range( *self.z_lims[event])
        #self.gals.set_area(self.cred_pixels[event], self.GWevents[event].nside)
        
        # Convolution with z errors
    
        rLow, rUp, nPoints = self._get_rLims(event)
        
        
        zUp = z_from_dLGW(rUp, H0,  Xi0, n=n)
        zLow = z_from_dLGW(rLow, H0,  Xi0, n=n)
        z_table = np.linspace(min(zLow-zLow/10, 0), zUp+zUp/10, 500)
        dLgw_table = dLGW(z_table, H0=H0, Xi0=Xi0, n=n)
        
        # interpolate the func z(dL) and evaluate on rGrid
        rGrid = np.linspace(rLow, rUp, nPoints)
        zGrid = np.interp( rGrid , dLgw_table, z_table).T
                
        pixels, z_weights = self.gals.get_inhom_contained(zGrid, self.GWevents[event].nside ) 
        #print('pixels shape: %s' %str(pixels.shape))
        l_weights = np.ones(z_weights.shape[0])
        
        my_skymap = np.array([self.GWevents[event].likelihood_px(r, pixels) for r in rGrid])
        #print('z_weights shape: %s' %str(z_weights.shape))
        #print('my_skymap shape: %s' %str(my_skymap.shape))
        L = np.sum(my_skymap.T*z_weights, axis=1)
        #print('L shape: %s' %str(L.shape))
        # Apply weights and normalize   
        #print('l_weights shape: %s' %str(l_weights.shape))
        LL =  (np.dot(L.T,l_weights))/l_weights.sum()
        
        return LL
    
    
    def _hom_lik(self, event, H0, Xi0, n=1.91):
        '''
        Computes likelihood homogeneous part for one event
        '''
        
        pass
    
    
    def _get_rLims(self, event, nsigma=3, minPoints=50):
        
        cred_pixels=self.cred_pixels[event]
        mu = self.GWevents[event].mu[cred_pixels]
        sigma = self.GWevents[event].sigma[cred_pixels]
        rl = mu-nsigma*sigma
        rl = np.where(rl>0., rl, 0.)
        #rl = np.nan_to_num(rl)
        ru =  mu+nsigma*sigma
        #ru = np.nan_to_num(ru)
        nPoints = minPoints*np.rint((mu.max()-mu.min())/sigma.max())
        
        return rl.min(), ru.max(), nPoints
    
    