#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:30:19 2020

@author: Michi
"""

####
# This module contains everything related to handling GW skymaps
####


import healpy as hp
import pandas as pd
import scipy.stats
from Xi0Stat.globals import *




class Skymap3D(object):
    

    def __init__(self, fname, 
                 nest=False,
                 verbose=False
                 ):
        
        self.verbose=verbose
        try:
            
            smap, header = hp.read_map(fname, field=range(4),
                                       h=True, nest=nest, verbose=False)
                 
        except IndexError:
            print('No parameters for 3D gaussian likelihood')
            smap = hp.read_map(fname, nest=nest, verbose=False)
            header=None
        
        try:
            self.event_name=dict(header)['OBJECT']
            if self.verbose:
                print('Event name: %s \n' %self.event_name)
        except KeyError:
            ename = fname.split('/')[-1].split('.')[0].split('_')[0]
            print('No event name in header for this event. Using name provided with filename %s ' %ename)
            self.event_name = ename
        
        self.npix = len(smap[0])
        self.nside = hp.npix2nside(self.npix)
        self.pixarea = hp.nside2pixarea(self.nside, degrees=False) # pixel area in square radians
        self.p = smap[0]
        self.head = header
        self.mu   = smap[1]
        self.sigma   = smap[2]
        self.norm   = smap[3]
        self.all_pixels = np.arange(self.npix)
        self.metadata = self._get_metadata()
        self.nest=nest
        
    
    def _get_metadata(self):
        O2metaPath = os.path.join(metaPath, 'GWTC-1-confident.csv')     
        try:
            df = pd.read_csv(O2metaPath)
            res = df[df['commonName']==self.event_name]
            #print(res.commonName.values)
            if res.shape[0]==0:
                print('No metadata found!')
                res=None
        except ValueError:
            print('No metadata available!')
            res=None
        return res
    
    
    def find_pix_RAdec(self, ra, dec):
        '''
        input: ra dec in degrees
        output: corresponding pixel with nside given by that of the skymap
        '''
        theta, phi = th_phi_from_ra_dec(ra, dec)
        
        # Note: when using ang2pix, theta and phi must be in rad 
        pix = hp.ang2pix(self.nside, theta, phi, nest=self.nest)
        return pix
    
    def find_pix(self, theta, phi):
        '''
        input: theta phi in rad
        output: corresponding pixel with nside given by that of the skymap
        '''

        pix = hp.ang2pix(self.nside, theta, phi, nest=self.nest)
        return pix

    def find_theta_phi(self, pix):
        '''
        input:  pixel
        output: (theta, phi)of pixel center in rad, with nside given by that of the skymap 
        '''
        return hp.pix2ang(self.nside, pix, nest=self.nest)
    
    
    def find_ra_dec(self, pix):
        '''
        input:  pixel ra dec in degrees
        output: (ra, dec) of pixel center in degrees, with nside given by that of the skymap 
        '''
        theta, phi = self.find_theta_phi(pix)
        ra, dec = ra_dec_from_th_phi(theta, phi)
        return ra, dec
    
    
    
    def find_event_coords(self):   
        return self.find_ra_dec(np.argmax(self.p))
          
    
    def dp_dr_cond(self, r, theta, phi):
        '''
        conditioned probability 
        p(r|Omega)
        
        input:  r  - GW lum distance in Mpc
                ra, dec
        output: eq. 2.19 , cfr 1 of Singer et al 2016
        '''
        pix = self.find_pix(theta, phi)
        return (r**2)*self.norm[pix]*scipy.stats.norm.pdf(x=r, loc=self.mu[pix], scale=self.sigma[pix]) 

    
    def dp_dr(self, r, theta, phi):
        '''
        p(r,Omega|data) = p(r|Omega)*p(Omega) : probability that a source is within pixel i and at a
        distance between r and r+dr
        
        p(Omega) = rho_i    Probability that source is in pixel i 
                            (Note that we don't divide rho by pixel area so this is not the prob density at Omega) 
        
        output: eq. 2 of singer et al 2016 . 
                This should be normalized to 1 when summing over pixels and integrating over r
        
        
        '''
        pix = self.find_pix(theta, phi)
        cond_p = self.dp_dr_cond(r, theta, phi)
        return cond_p*self.p[pix] #/self.pixarea
    
    
    def likelihood(self, r, theta, phi):
        '''
        Eq. 2.18 
        Likelihood given r , theta, phi ( theta, phi in rad)
        p(data|r,Omega) = p(r,Omega|data) * p(r) 
        p(r) = r^2
        
        '''
        #pix = self.find_pix(ra, dec)
        #LL = self.p[pix]*self.norm[pix]*scipy.stats.norm.pdf(x=r, loc=self.mu[pix], scale=self.sigma[pix])  #  = self.dp_dr(r, ra, dec)/r**2
        return self.likelihood_px(r, self.find_pix(theta, phi))
    
    def likelihood_px(self, r, pix):
        '''
        Eq. 2.18
        Likelihood given pixel. Note: to avoid negative values, we truncate the gaussian at zero.
        L(data|r,Omega_i)
          
        p(r,Omega_i|data) = L(data|r,Omega_i) * p(r) 
        p(r) = r^2
        '''
        myclip_a=0
        myclip_b=np.infty
        a, b = (myclip_a - self.mu[pix]) / self.sigma[pix], (myclip_b - self.mu[pix]) / self.sigma[pix]
        return  self.p[pix]*self.norm[pix]*scipy.stats.truncnorm(a=a, b=b, loc=self.mu[pix], scale=self.sigma[pix]).pdf(r)
        #scipy.stats.norm.pdf(x=r, loc=self.mu[pix], scale=self.sigma[pix])
    
    
    def p_r(self, r):  
        '''
        Posterior on lum. distance p(r|data) 
        marginalized over Omega
        To be compared with posterior chains 
        '''
        return sum(self.p*self.norm*scipy.stats.norm(loc=self.mu, scale=self.sigma).pdf(r) )*r**2
    
    def p_om(self, theta, phi):
        '''
        p(Omega)
        '''
        return self.p[self.find_pix(theta, phi)]
    
    def area_p(self, pp=0.9):
        ''' Area of pp% credible region '''
        i = np.flipud(np.argsort(self.p))
        sorted_credible_levels = np.cumsum(self.p[i])
        credible_levels = np.empty_like(sorted_credible_levels)
        credible_levels[i] = sorted_credible_levels
        #from ligo.skymap.postprocess import find_greedy_credible_levels
        #credible_levels = find_greedy_credible_levels(self.p)
        
        return np.sum(credible_levels <= pp) * hp.nside2pixarea(self.nside, degrees=True)
    
    
    def _get_credible_region_pth(self, level=0.99):
        '''
        Finds value minskypdf of rho_i that bouds the x% credible region , with x=level
        Then to select pixels in that region: self.all_pixels[self.p>minskypdf]
        '''
        prob_sorted = np.sort(self.p)[::-1]
        prob_sorted_cum = np.cumsum(prob_sorted)
        # find index of array which bounds the self.area confidence interval
        idx = np.searchsorted(prob_sorted_cum, level)
        minskypdf = prob_sorted[idx] #*skymap.npix
        
        #self.p[self.p]  >= minskypdf       
        return minskypdf
    
    def get_credible_region_pixels(self, level=0.99):
        return self.all_pixels[self.p>self._get_credible_region_pth(level=level)]
    
    
    def likelihood_in_credible_region(self, r, level=0.99, verbose=False):
        '''
        Returns likelihood for all the pixels in the x% credible region at distance r
        x=level
        '''
        cArea_idxs = self.get_credible_region_pixels(level=level)
        LL = self.likelihood_px(r, cArea_idxs)
        if verbose:
            print('Max GW likelihood at dL=%s Mpc : %s' %(r,LL.max()))
            print('Pix of max GW likelihood = %s' %cArea_idxs[LL.argmax()])
            print('RA, dec of max GW likelihood at dL=%s Mpc: %s' %(r,self.find_ra_dec(cArea_idxs[LL.argmax()])))
        return LL
    
 
    def d_max(self, SNR_ref=8):
        '''
        Max GW luminosity distance at which the evend could be seen, 
        assuming its SNR and a threshold SNR_ref:
        d_max = d_obs*SNR/SNR_ref
        
        d_obs is the posterior mean quoted in the GW metadata
        '''
        try:
            d_obs = self.metadata['luminosity_distance'].values[0]
            SNR = self.metadata['network_matched_filter_snr'].values[0]
            #print('using d_obs and SNR from metadata')
        except IndexError:
            print('SNR for this event not available! Computing d_max with SNR=%s' %SNR_ref)
            d_obs = np.float(dict(self.head)['DISTMEAN'])
            SNR = SNR_ref
        #std_quoted = np.float(dict(self.head)['DISTSTD'])
        d_max = d_obs*SNR/SNR_ref
        
        return d_max
        
    
    def get_zlims(self, H0max=220, H0min=20, Xi0max=3, Xi0min=0.2, n=1.91, verbose=False):  
        '''
        Computes the possible range in redshift for the event given the prior range for H0 and XI0
        '''
        
        if verbose:
            print('Computing range in redshift for all priors...')
        grid=[]
        for H0i in [H0min, H0max]:
            for Xi0i in [Xi0min, Xi0max]:
                grid.append([H0i, Xi0i]) 
        zs = np.array([self._get_minmaxz(*vals,n=n, verbose=verbose) for vals in grid ])
        
        return zs.min(), zs.max()
    
    
    def _get_minmaxz(self, H0, Xi0, n=1.91, std_number=3, verbose=False):
        '''
        Upper and lower limit in redshift to search for given H0 or Xi0
        '''
        
        # Get mu and sigma for dLGW
        map_val = np.float(dict(self.head)['DISTMEAN'])
        up_lim = np.float(dict(self.head)['DISTSTD'])
        low_lim=-up_lim
        dlow = max(map_val+std_number*low_lim,0)
        dup=map_val+std_number*up_lim
        if verbose:
            print('Position: %s +%s %s'%(map_val, up_lim, low_lim))
        z1 = z_from_dLGW(dlow, H0, Xi0, n=n) 
        z2 = z_from_dLGW(dup,  H0, Xi0, n=n)
        if verbose:
            print('H0, Xi0: %s, %s' %(H0, Xi0))
            print('lower limit to search: d_L = %s Mpc, z=%s' %(dlow,z1))
            print('upper limit to search:d_L = %s Mpc, z=%s' %(dup, z2))
 
        minmax_z = max(min(z1,z2), 0), max(z1,z2) 
    
        return minmax_z 
            
        
    
    
def get_all_events(loc='data/GW/O2/', subset=True, subset_names=['GW170817',], 
                   verbose=False
               ):
    '''
    Returns dictionary with all skymaps in the folder loc.
    If subset=True, gives skymaps only for the event specified by subset_names
    
    '''
    from os import listdir
    from os.path import isfile, join
    sm_files = [f for f in listdir(loc) if ((isfile(join(loc, f))) & (f!='.DS_Store'))]    
    ev_names = [fname.split('_')[0]  for fname in sm_files]
    if subset:
        ev_names = [e for e in ev_names if e in subset_names]
        sm_files = [e+'_skymap.fits' for e in ev_names]
    if verbose:
        print('--- GW events:')
        print(ev_names)
        print('Reading skymaps....')
    all_events = {fname.split('_')[0]: Skymap3D(loc+fname, nest=False) for fname in sm_files}
    return all_events
    
