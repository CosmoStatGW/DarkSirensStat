#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:51:19 2020

@author: Michi
"""

####
# This module contains a class to handle GW-galaxy correlation and compute the likelihood
####
import numpy as np
from Xi0Stat.globals import *


class GWgal(object):
    
    def __init__(self, GalCompleted, GWevents,):
        
        self.gals = GalCompleted
        self.GWevents = GWevents
        
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
        '''
        
        
        
        z_table = np.linspace(0, 10, 10000)
        dLgw_table = dL_GW(z_table, H0=H0, Xi0=Xi0, n=n)
        
        rLow= self._get_rLow(event)
        rUp=self._get_rUp(event)
        rGrid = np.linspace(rLow, rUp, 50)
        
        # interpolate the func z(dL) and evaluate on rGrid
        zGrid = np.interp( rGrid , dLgw_table, z_table).T
        
        pixels, weights = self.gals.get_inhom_contained(zGrid, self.nside ) 
        
        my_skymap = self.GWevents[event].likelihood_px(rGrid, pixels)
        L = np.sum(my_skymap.T*weights, axis=1)
        
        return L
    
    def _hom_lik(self, event, H0, Xi0, n=1.91):
        '''
        Computes likelihood homogeneous part for one event
        '''
        
        pass
    
    