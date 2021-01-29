#
#    Copyright (c) 2021 Andreas Finke <andreas.finke@unige.ch>,
#                       Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by a modified BSD
#    license that can be found in the LICENSE file.


####
# This module contains a class to handle GW-galaxy correlation and compute the likelihood
####
from globals import *
import pandas as pd
from copy import deepcopy



class GWgal(object):
    
    def __init__(self, GalCompleted, GWevents, 
                 eventSelector,
                 MC = True, nHomSamples=1000, 
                 galRedshiftErrors = True, 
                 zR=zRglob,
                 verbose=False):
        
        self.eventSelector=eventSelector
        self.gals = GalCompleted
        self.GWevents = GWevents
        self.selectedGWevents= deepcopy(GWevents)

        self._galRedshiftErrors = galRedshiftErrors
        self.verbose=verbose
        
        self.nHomSamples = nHomSamples
        self.MC=MC
        self.zR=zR
        self.nGals={}
        
        
        self._get_avgPcompl()
        self._select_events()
        
        for eventName in GWevents.keys():
            self.select_gals_event(eventName)
            self.nGals[eventName] = self.gals.count_selection()
        # Note on the generalization. Eventually we should have a dictionary
        # {'GLADE': glade catalogue, 'DES': ....}
        # and a dictionary {'GW event': 'name of the catalogue to use'}
        # The function _inhom_lik will use the appropriate catalogue according to
        # this dictionary
        
        # Completeness needs a name or something to know if we use 
        # multiplicative, homogeneous or no completion
        #if self.verbose:
        #    print('\n --- GW events: ')
        #    for event in GWevents.keys():
        #        print(event)
    
    
    def _select_events(self):
        
            self.selectedGWevents = { eventName:self.GWevents[eventName] for eventName in self.GWevents.keys() if self.eventSelector.is_good_event(self.GWevents[eventName]) }
        #print('Selected GW events with Pc_Av>%s or Pc_event>%s. Events: %s' %(completnessThreshAvg, completnessThreshCentral, str(list(self.selectedGWevents.keys()))))
            if self.verbose:
                print('Selected GW events: %s' %( str(list(self.selectedGWevents.keys()))))
    
    
    def select_gals_event(self,eventName):
        self.gals.select_area(self.GWevents[eventName].selected_pixels, self.GWevents[eventName].nside)
        self.gals.set_z_range_for_selection( *self.GWevents[eventName].get_z_lims())
    
    def _get_summary(self):
        
        self.summary = pd.DataFrame.from_dict({'name': [self.GWevents[eventName].event_name for eventName in self.GWevents.keys()],
         'Omega_degSq': [self.GWevents[eventName].area() for eventName in self.GWevents.keys()],
         'dL_Mpc': [self.GWevents[eventName].dL for eventName in self.GWevents.keys()],
        'dLlow_Mpc':[self.GWevents[eventName].dLmin for eventName in self.GWevents.keys()],
        'dLup_Mpc':[self.GWevents[eventName].dLmax for eventName in self.GWevents.keys()],
        'z_event':[self.GWevents[eventName].zfiducial for eventName in self.GWevents.keys()],
        'zLow':[self.GWevents[eventName].zmin for eventName in self.GWevents.keys()],
        'zUp':[self.GWevents[eventName].zmax for eventName in self.GWevents.keys()],
         'Vol_mpc3':[self.GWevents[eventName].volCom for eventName in self.GWevents.keys()],
         'nGal':[self.nGals[ eventName] if eventName in self.selectedGWevents.keys() else '--' for eventName in self.GWevents.keys()],
         'Pc_Av': [self.PcAv[eventName] for eventName in self.GWevents.keys()],
         'Pc_event': [self.PEv[eventName] for eventName in self.GWevents.keys()]})
        
        
        
        
        
        
        
    def _get_avgPcompl(self):
        if self.verbose:
            print('Computing <P_compl>...')
        PcAv={}
        PEv = {}
        #from scipy.integrate import quad
        for eventName in self.GWevents.keys():
            #self.GWevents[eventName].adap_z_grid(H0GLOB, Xi0Glob, nGlob, zR=self.zR)
            zGrid = np.linspace(self.GWevents[eventName].zmin, self.GWevents[eventName].zmax, 100)
            
            if self.GWevents[eventName].selected_pixels.size==0:
                #Pcomp=np.zeros(zGrid.shape)
                PcAv[eventName] = 0.
            else:
                Pcomp = np.array([self.gals.total_completeness( *self.GWevents[eventName].find_theta_phi(self.GWevents[eventName].selected_pixels), z).sum() for z in zGrid])
                vol = self.GWevents[eventName].areaRad*np.trapz(cosmoglob.differential_comoving_volume(zGrid).value, zGrid) #quad(lambda x: cosmoglob.differential_comoving_volume(x).value, self.GWevents[eventName].zmin,  self.GWevents[eventName].zmax)
                _PcAv = np.trapz(Pcomp*cosmoglob.differential_comoving_volume(zGrid).value, zGrid)*self.GWevents[eventName].pixarea/vol
                PcAv[eventName] = _PcAv
            
            
            _PEv = self.gals.total_completeness( *self.GWevents[eventName].find_event_coords(polarCoords=True), self.GWevents[eventName].zfiducial)
            PEv[eventName] = _PEv
            if self.verbose:
                print('<P_compl> for %s = %s; Completeness at (z_event, Om_event): %s' %(eventName, np.round(_PcAv, 3), np.round(_PEv, 3)))        
        self.PcAv = PcAv
        self.PEv = PEv
        
    
    
    def get_lik(self, H0s, Xi0s, n=nGlob):
        '''
        Computes likelihood for all events
        Returns dictionary {event_name: L_cat, L_comp }
        '''
        ret = {}
        H0s = np.atleast_1d(H0s)
        Xi0s = np.atleast_1d(Xi0s)
        for eventName in self.selectedGWevents.keys():
            if self.verbose:
                print('-- %s' %eventName)
            
            #self.gals.select_area(self.selectedGWevents[eventName].selected_pixels, self.selectedGWevents[eventName].nside)
            #self.nGals[eventName] = self.gals.set_z_range_for_selection( *self.selectedGWevents[eventName].get_z_lims(), return_count=True)
            self.select_gals_event(eventName)
            
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
        Output:
        '''
        
        if self._galRedshiftErrors:
        
            # Convolution with z errors
            
            rGrid = self._get_rGrid(eventName, minPoints=20)

            zGrid = z_from_dLGW_fast(rGrid, H0=H0, Xi0=Xi0, n=n)
            
            pixels, weights = self.gals.get_inhom_contained(zGrid, self.selectedGWevents[eventName].nside )
            
            my_skymap = self.selectedGWevents[eventName].likelihood_px(rGrid[np.newaxis, :], pixels[:, np.newaxis])
         
            #LL = np.sum(skymap*weights)
             
        else: # use Diracs
            
            pixels, zs, weights =  self.gals.get_inhom(self.selectedGWevents[eventName].nside)
            
            rs = dLGW(zs, H0=H0, Xi0=Xi0, n=n)
            
            my_skymap = self.selectedGWevents[eventName].likelihood_px(rs, pixels)
            
        LL = np.sum(my_skymap*weights)
        
        return LL
    
    def _hom_lik(self, eventName, H0, Xi0, n):
        
        if self.MC: 
            return self._hom_lik_MC(eventName, H0, Xi0, n)
        else: 
            return self._hom_lik_trapz(eventName, H0, Xi0, n)
        
        
    
    def _hom_lik_trapz(self, eventName, H0, Xi0, n):
        
        zGrid = self.selectedGWevents[eventName].adap_z_grid(H0, Xi0, n, zR=self.zR)
        
        #self.gals.eval_hom(theta, phi, z) #glade._completeness.get( *myGWgal.selectedGWevents[ename].find_theta_phi(pxs), z)
        
        pxs = self.selectedGWevents[eventName].get_credible_region_pixels()
        th, ph = self.selectedGWevents[eventName].find_theta_phi(pxs)
        
        integrand_grid = np.array([ j(z)*(self.gals.eval_hom(th, ph, z, MC=False))*self.selectedGWevents[eventName].likelihood_px( dLGW(z, H0, Xi0, n), pxs) for z in zGrid])
        
        integral = np.trapz(integrand_grid.sum(axis=1), zGrid)
        den = (70/clight)**3
        LL = integral*self.selectedGWevents[eventName].pixarea/den
        
        return LL
    
    
    def _hom_lik_MC(self, eventName, H0, Xi0, n):
        '''
        Computes likelihood homogeneous part for one event
        '''
        
        theta, phi, r = self.selectedGWevents[eventName].sample_posterior(nSamples=self.nHomSamples)
        
        z = z_from_dLGW_fast(r, H0=H0, Xi0=Xi0, n=n)
        
        # the prior is nbar in comoving volume, so it transforms if we integrate over D_L^{gw}
        # nbar D_com^2 d D_com = nbar D_com^2 (d D_com/d D_L^{gw}) d D_L^{gw}
        
        # we put a D_L^{gw}^2 into sampling from the posterior instead from the likelihood, and divide the jacobian by it.
       
        jac = dVdcom_dVdLGW(z, H0=H0, Xi0=Xi0, n=n)
         
        # MC integration
        
        LL = (H0/70)**3*np.mean(jac*self.gals.eval_hom(theta, phi, z))
        
        return LL
    
    
    def _get_rGrid(self, eventName, minPoints=50):
    
        lower, upper = self.selectedGWevents[eventName].dLmin, self.selectedGWevents[eventName].dLmax,
        
        nPoints = np.int(minPoints*(upper-lower)/self.selectedGWevents[eventName].sigmadL)
        
        return np.linspace(lower, upper, nPoints)
    
    
