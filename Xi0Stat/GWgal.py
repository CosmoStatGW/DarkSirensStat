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
        self.cred_pixels = {event: self.GWevents[event].get_credible_region_pixels(level=credible_level) for event in self.GWevents}
        self.z_lims = {event: self.GWevents[event].get_zlims() for event in self.GWevents}
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
        
            self.gals.select_area(self.cred_pixels[eventName], self.GWevents[eventName].nside)
  
            self.gals.set_z_range_for_selection( *self.z_lims[eventName])
            
            Linhom = np.ones((H0s.size, Xi0s.size))
            Lhom   = np.ones((H0s.size, Xi0s.size))
        
            for i in np.arange(H0s.size):
            
                for j in np.arange(Xi0s.size):
           
                    Linhom[i,j] = self._inhom_lik(eventName=eventName, H0=H0s[i], Xi0=Xi0s[j], n=n)
                    
                    Lhom[i,j] = self._hom_lik(eventName=eventName, H0=H0s[i], Xi0=Xi0s[j], n=n)

            ret[eventName] = (np.squeeze(Linhom), np.squeeze(Lhom))
            
        return ret
     
    
    
#    def get_hom_lik(self, H0, Xi0, n=nGlob):
#        '''
#        Computes likelihood with homogeneous part for all events.
#        Returns dictionary {event_name: L_hom }
#        '''
#        return {event: self._hom_lik(event, H0=H0, Xi0=Xi0, n=n) for event in self.GWevents}
            
    
    
    def _inhom_lik(self, eventName, H0, Xi0, n):
        '''
        Computes likelihood with p_cat for one event
        Output: np array of dim (N. galaxies in 99% credible region, 1)
        '''
        
        
        if self._galRedshiftErrors:
        
            # Convolution with z errors
            
            rLow, rUp, nPoints = self._get_rLims(eventName)
            rGrid = np.linspace(rLow, rUp, nPoints)
#            zUp = z_from_dLGW(rUp, H0,  Xi0, n=n)
#            zLow = z_from_dLGW(rLow, H0,  Xi0, n=n)
#            z_table = np.linspace(min(zLow-zLow/10, 0), zUp+zUp/10, 500)
#            dLgw_table = dLGW(z_table, H0=H0, Xi0=Xi0, n=n)
#            # interpolate the func z(dL) and evaluate on rGrid
            
#            zGrid = np.interp( rGrid , dLgw_table, z_table).T

            zGrid = z_from_dLGW_fast(rGrid, H0=H0, Xi0=Xi0, n=n)
            
            pixels, weights = self.gals.get_inhom_contained(zGrid, self.GWevents[eventName].nside )
            
            #print('pixels shape: %s' %str(pixels.shape))
            #l_weights = np.ones(z_weights.shape[0])
            
            #my_skymap = np.array([self.GWevents[event].likelihood_px(r, pixels) for r in rGrid]).T
    
            my_skymap = self.GWevents[eventName].likelihood_px(rGrid[np.newaxis, :], pixels[:, np.newaxis])
            
            #print('z_weights shape: %s' %str(z_weights.shape))
            #print('my_skymap shape: %s' %str(my_skymap.shape))
            LL = np.sum(my_skymap*weights)
            #print('L shape: %s' %str(L.shape))
            # Apply weights and normalize
            #print('l_weights shape: %s' %str(l_weights.shape))
            #LL =  (np.dot(L.T,l_weights))/l_weights.sum()
            
        else: # use Diracs
            
            pixels, zs, weights =  self.gals.get_inhom(self.GWevents[eventName].nside)
            
            rs = dLGW(zs, H0=H0, Xi0=Xi0, n=n)
            
            my_skymap = self.GWevents[eventName].likelihood_px(rs, pixels)
            
            LL = np.sum(my_skymap*weights)
        
        return LL
    
    
    def _hom_lik(self, eventName, H0, Xi0, n):
        '''
        Computes likelihood homogeneous part for one event
        '''
        nSamples = 1000
        
        theta, phi, r = self.GWevents[eventName].sample(nSamples=nSamples)
        
        z = z_from_dLGW_fast(r, H0=H0, Xi0=Xi0, n=n)
        
        # the prior is nbar in comoving volume, so it transforms if we integrate over D_L^{gw}
        # nbar D_com^2 d D_com = nbar D_com^2 (d D_com/d D_L^{gw}) d D_L^{gw}
       
        jac = dVdcom_dVdLGW(z, H0=H0, Xi0=Xi0, n=n)
         
        # MC integration
        
        LL = np.sum(jac*self.gals.eval_hom(theta, phi, z))/nSamples
        
        return LL
    
    
    def _get_rLims(self, eventName, nsigma=3, minPoints=50):
        
        cred_pixels=self.cred_pixels[eventName]
        mu = self.GWevents[eventName].mu[cred_pixels]
        sigma = self.GWevents[eventName].sigma[cred_pixels]
        rl = mu-nsigma*sigma
        rl = np.where(rl>0., rl, 0.)
        #rl = np.nan_to_num(rl)
        ru =  mu+nsigma*sigma
        #ru = np.nan_to_num(ru)
        nPoints = minPoints*np.int((mu.max()-mu.min())/sigma.max())
        
        return rl.min(), ru.max(), nPoints
    
    
