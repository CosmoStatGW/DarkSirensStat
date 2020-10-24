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
    
    def __init__(self, GalCompleted, GWevents, credible_level=0.99, galRedshiftErrors = True, verbose=False):
        
        self.gals = GalCompleted
        self.GWevents = GWevents

        self._galRedshiftErrors = galRedshiftErrors
        self.verbose=verbose
        
        
        # Note on the generalization. Eventually we should have a dictionary
        # {'GLADE': glade catalogue, 'DES': ....}
        # and a dictionary {'GW event': 'name of the catalogue to use'}
        # The function _inhom_lik will use the appropriate catalogue according to
        # this dictionary
        
        # Completeness needs a name or something to know if we use 
        # multiplicative, homogeneous or no completion
        if self.verbose:
            print('\n --- GW events: ')
            for event in GWevents.keys():
                print(event)
            
    
    
    def get_lik(self, H0s, Xi0s, n=nGlob):
        '''
        Computes likelihood with p_cat for all events
        Returns dictionary {event_name: L_cat }
        '''
        ret = {}
        H0s = np.atleast_1d(H0s)
        Xi0s = np.atleast_1d(Xi0s)
        
        for eventName in self.GWevents.keys():
        
            self.gals.select_area(self.GWevents[eventName].selected_pixels, self.GWevents[eventName].nside)
  
            self.gals.set_z_range_for_selection( *self.GWevents[eventName].get_z_lims())
            
            Linhom = np.ones((H0s.size, Xi0s.size))
            Lhom   = np.ones((H0s.size, Xi0s.size))
        
            for i in np.arange(H0s.size):
            
                for j in np.arange(Xi0s.size):
                    
           
                    Linhom[i,j] = self._inhom_lik(eventName=eventName, H0=H0s[i], Xi0=Xi0s[j], n=n)
                    
                    Lhom[i,j] = self._hom_lik(eventName=eventName, H0=H0s[i], Xi0=Xi0s[j], n=n)

            ret[eventName] = (np.squeeze(Linhom), np.squeeze(Lhom))
            
        return ret
     
    
    def _inhom_lik(self, eventName, H0, Xi0, n):
        '''
        Computes likelihood with p_cat for one event
        Output: np array of dim (N. galaxies in 99% credible region, 1)
        '''
        
        if self._galRedshiftErrors:
        
            # Convolution with z errors
            
            rGrid = self._get_rGrid(eventName, nsigma=3, minPoints=50)

            zGrid = z_from_dLGW_fast(rGrid, H0=H0, Xi0=Xi0, n=n)
            
            pixels, weights = self.gals.get_inhom_contained(zGrid, self.GWevents[eventName].nside )
            
            skymap = self.GWevents[eventName].likelihood_px(rGrid[np.newaxis, :], pixels[:, np.newaxis])
         
            LL = np.sum(skymap*weights)
             
        else: # use Diracs
            
            pixels, zs, weights =  self.gals.get_inhom(self.GWevents[eventName].nside)
            
            rs = dLGW(zs, H0=H0, Xi0=Xi0, n=n)
            
            my_skymap = self.GWevents[eventName].likelihood_px(rs, pixels)
            
            LL = np.sum(my_skymap*weights)
            
            #print(weights.shape, weights)
        
        return LL
    
    
    def _hom_lik(self, eventName, H0, Xi0, n):
        '''
        Computes likelihood homogeneous part for one event
        '''
        nSamples = 400
        
        theta, phi, r = self.GWevents[eventName].sample_posterior(nSamples=nSamples)
        
        z = z_from_dLGW_fast(r, H0=H0, Xi0=Xi0, n=n)
        
        # the prior is nbar in comoving volume, so it transforms if we integrate over D_L^{gw}
        # nbar D_com^2 d D_com = nbar D_com^2 (d D_com/d D_L^{gw}) d D_L^{gw}
        
        # we put a D_L^{gw}^2 into sampling from the posterior instead from the likelihood, and divide the jacobian by it.
       
        jac = dVdcom_dVdLGW_divided_by_dLGWsq(z, H0=H0, Xi0=Xi0, n=n)
         
        # MC integration
        
        LL = (H0/70)**3*np.mean(jac*self.gals.eval_hom(theta, phi, z))
        
        return LL
    
    
    def _get_rGrid(self, eventName, nsigma=3, minPoints=30):
    
        meanmu, lower, upper, meansig = self.GWevents[eventName].find_r_loc(std_number = nsigma)
        
        cred_pixels = self.GWevents[eventName].selected_pixels
        
        nPoints = np.int(minPoints*(upper-lower)/meansig)
        
        return np.linspace(lower, upper, nPoints)
    
    
