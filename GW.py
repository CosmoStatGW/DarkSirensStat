#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 12:21:35 2020

@author: Michi
"""

import numpy as np
import scipy.stats
#from astropy.io import fits
import healpy as hp
import os
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.interpolate import  UnivariateSpline #interp1d,
#from scipy.optimize import fsolve
#import scipy.interpolate as interpolate
import pandas as pd
from utils import *



class skymap_3D(object):

    def __init__(self, fname, nest=True,
                 cosmo='FlatLCDM', H0=70, Om0=0.27,
                 meta_path = '/Users/Michi/Dropbox/Local/Physics_projects/statistical_method_schutz_data_local/GWs/GWTC-1-confident.csv'):
        try:
            
            smap, header = hp.read_map(fname, field=range(4),
                                       h=True, nest=nest)
                 
        except IndexError:
            print('No parameters for 3D gaussian likelihood')
            smap = hp.read_map(fname, nest=nest)
            header=None
        
        self.Om0 = Om0
        if cosmo=='FlatLCDM':
            from astropy.cosmology import FlatLambdaCDM
            self.cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        else: 
            self.cosmo=cosmo
        try:
            self.event_name=dict(header)['OBJECT']
            print('Event name: %s \n' %self.event_name)
        except KeyError:
            ename = fname.split('/')[-1].split('.')[0].split('_')[0]
            print('No event name in header for this event. Using name provided with filename %s ' %ename)
            self.event_name = ename
        self.p = smap[0]
        self.head = header
        self.mu   = smap[1]
        self.sigma   = smap[2]
        self.norm   = smap[3]
        self.npix = len(self.p)
        self.nside = hp.npix2nside(self.npix)
        self.pixarea = hp.nside2pixarea(self.nside, degrees=False) # pixel area in square radians
        self.all_pixels = np.arange(self.npix)
        #self.credible_levels = None
        self.metadata = self._get_metadata(meta_path)
    
    
    def find_pix(self, ra, dec):
        '''
        ra dec in degrees
        '''
        theta = 0.5 * np.pi - np.deg2rad(dec)
        phi = np.deg2rad(ra)
        pix = hp.ang2pix(self.nside, theta, phi)
        return pix
    
    def _get_metadata(self, meta_path):
        try:
            df = pd.read_csv(meta_path)
            res = df[df['commonName']==self.event_name]
            print(res.commonName.values)
            if res.shape[0]==0:
                print('No metadata found!')
                res=None
        except ValueError:
            print('No metadata available!')
            res=None
        return res
    
    def find_event_coords(self):   
        return self.find_ra_dec(np.argmax(self.p))
    
    def find_ra_dec(self, pix):
        theta, phi = hp.pix2ang(self.nside, pix)
        ra = np.rad2deg(phi)
        dec = np.rad2deg(0.5 * np.pi - theta)
        return ra, dec
       
    
    def dp_dr_cond(self, r, ra, dec):
        '''
        conditioned probability 
        p(r|Omega)
        '''
        pix = self.find_pix(ra, dec)
        return (r**2)*self.norm[pix]*scipy.stats.norm.pdf(x=r, loc=self.mu[pix], scale=self.sigma[pix]) 

    
    def dp_dr(self, r, ra, dec):
        '''
        posterior in D_L, Omega
        p(r,Omega|data)= p(r|Omega)*p(Omega) 
        p(Omega) = rho_i / pixarea
        '''
        pix = self.find_pix(ra, dec)
        cond_p = self.dp_dr_cond(r, ra, dec)
        return cond_p*self.p[pix]/self.pixarea
    
    
    def likelihood(self, r, ra, dec, eps=1e-20):
        '''
        Likelihood (to be used in the computation of the posterior!)
        p(data|r,Omega) = p(r,Omega|data) * p(r) 
        p(r) = r^2
        ra, dec in degrees
        '''
        pix = self.find_pix(ra, dec)
        #LL = self.p[pix]*self.norm[pix]*scipy.stats.norm.pdf(x=r, loc=self.mu[pix], scale=self.sigma[pix])  #self.dp_dr(r, ra, dec)/r**2
        return  self.likelihood_px( r, pix) #LL#np.where(LL>eps,LL,eps )
    
    
    
    def likelihood_px(self, r, pix):
        '''
        Likelihood (to be used in the computation of the posterior!)
        L(data|r,Omega_i)
        
     
        p(r,Omega_i|data) = L(data|r,Omega_i) * p(r) 
        p(r) = r^2
        '''
        dp_dr_cond = self.norm[pix]*scipy.stats.norm(loc=self.mu[pix], scale=self.sigma[pix]).pdf(r)
        dp_dr = dp_dr_cond*self.p[pix]#/self.pixarea
        
        return  dp_dr
    
    
           
    def p_r(self, r):  
        '''
        Posterior on lum. distance p(r|data) 
        marginalized over Omega
        To be compared with posterior chains 
        '''
        return sum(self.p*self.norm*scipy.stats.norm(loc=self.mu, scale=self.sigma).pdf(r) )*r**2
    
    def p_om(self, ra, dec):
        '''
        p(Omega), ra and dec in degrees
        '''
        return self.p[self.find_pix(ra, dec)]
    
    def area_p(self, pp=0.9):
        # Area of pp% credible region
        from ligo.skymap.postprocess import find_greedy_credible_levels
        credible_levels = find_greedy_credible_levels(self.p)
        return np.sum(credible_levels <= pp) * hp.nside2pixarea(self.nside, degrees=True)
        
 
    def get_credible_region_idx(self, level=0.99):
        prob_sorted = np.sort(self.p)[::-1]
        prob_sorted_cum = np.cumsum(prob_sorted)
        # find index of array which bounds the self.area confidence interval
        idx = np.searchsorted(prob_sorted_cum, level)
        minskypdf = prob_sorted[idx] #*skymap.npix
        
        #self.p[self.p]  >= minskypdf       
        return minskypdf
    
    
    def likelihood_in_credible_region(self, r, level=0.99, Verbose=False):
        
        p_th = self.get_credible_region_idx(level=0.99)
        cArea_idxs = self.all_pixels[self.p>p_th]
        LL = self.likelihood_px(r, cArea_idxs)
        if Verbose:
            print('Max GW likelihood = %s' %LL.max())
        return LL
    
    def _get_cosmo(self, H0=None):
        if H0 is None:
            #print('Using self.cosmo')
            cosmo = self.cosmo
        else:
            #print('Using cosmo with H0=%s' %H0)
            from astropy.cosmology import FlatLambdaCDM
            cosmo = FlatLambdaCDM(H0=H0, Om0=self.Om0)
        return cosmo
    
    
    def z_max(self, Xi0=1, dL=None, n=1.91, SNR_ref=8, H0=None):
        cosmo = self._get_cosmo(H0=H0)
        #print('z_max call. H0=%s' %cosmo.H0)
        if H0 is not None:
            Xi0=1
        if dL is None:
            d_max = self.d_max(SNR_ref)
        else:
            d_max = dL
        #print('z_max call. d_max=%s' %d_max)
        res = z(d_max, cosmo, Xi0=Xi0, n=n, H0=H0)
        return res
    
    def d_max(self, SNR_ref=8):
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
        
    def beta(self, cat=None, Xi0=1, H0=None, dL=None, scheme='cat', 
             band='B', which_z='z_corr', lum_weighting=False):
        
        if scheme=='uniform':
            return self._beta_uniform( Xi0=Xi0, H0=H0, dL=dL)
        elif scheme=='cat':
            if cat is None:
                raise ValueError('Provide valid catalogue to compute beta')
            return self._beta_cat( cat=cat, Xi0=Xi0, H0=H0, dL=dL, band=band, which_z=which_z, lum_weighting=lum_weighting)
        
     
    def _beta_cat(self, cat, Xi0=1, H0=None, 
                  dL=None, band='B', which_z='z_corr', 
                  lum_weighting=False, **params):  
        
        #cosmo = self._get_cosmo(H0=H0)

        z_lim = self.z_max(Xi0=Xi0, dL=dL, H0=H0)
        
        df = cat[ cat[which_z]< z_lim ]
        #df_norm = cat[ cat[which_z]< self.z_max(Xi0=1, dL=dL, H0=70) ]
        df_norm=cat
        
        if band is not None and lum_weighting:
                             
            l_name=band+'_Lum'
            num = df[l_name].sum()
            norm=df_norm[l_name].sum()
        else:
            num = df.shape[0]
            norm = df_norm.shape[0]
        
        return num/norm
    
     
    def _beta_uniform(self, Xi0=1, H0=None, dL=None, **params):
        cosmo = self._get_cosmo( H0=H0)
        from scipy.integrate import quad        
        norm = quad(lambda x: cosmo.differential_comoving_volume(x).value, 0, self.z_max(Xi0=1, dL=dL, H0=70, **params) )[0]
        #norm = quad(lambda x: cosmo.differential_comoving_volume(x).value, 0, 6.48 )[0]

        num = quad(lambda x:  cosmo.differential_comoving_volume(x).value, 0, self.z_max(Xi0=Xi0, dL=dL, H0=H0, **params)  )[0]
        return num/norm
    
    def _get_minmax_d(self, Xi0=1, H0=None, std_number=3, n=1.91, 
                     Verbose=False, position_val='metadata'):
        #print('_get_minmax_d std_number: %s ' %std_number)
        cosmo = self._get_cosmo( H0=H0)
        if Verbose:
            print('Using H0=%s' %cosmo.H0)
        
        #print('using position val %s' %position_val)
        if position_val=='metadata':
            
            map_val = self.metadata['luminosity_distance'].values[0]
            up_lim = self.metadata['luminosity_distance_upper'].values[0]
            low_lim = self.metadata['luminosity_distance_lower'].values[0]
            
            dlow = max(map_val+low_lim,0)
            dup=map_val+up_lim
        elif position_val=='header':
            map_val = np.float(dict(self.head)['DISTMEAN'])
            up_lim = np.float(dict(self.head)['DISTSTD'])
            #print(up_lim)
            low_lim=-up_lim
            dlow = max(map_val+std_number*low_lim,0)
            dup=map_val+std_number*up_lim
        elif position_val=='posterior': 
            map_val, up_lim, low_lim = self.get_CI_from_posterior_sample(low_prob=0.05, up_prob=0.90)
            dlow=map_val-low_lim
            dup=map_val+up_lim
        
        if Verbose:
            print('Position: %s +%s %s'%(map_val, up_lim, low_lim))
        return map_val, dlow, dup 
        
    
    def _get_minmaxz(self, Xi0=1, H0=None, std_number=3, n=1.91, 
                     Verbose=False, position_val='metadata'):
        
        
        #print('_get_minmaxz std_number: %s ' %std_number)
        #print('H0 _get_minmaxz : %s ' %str(H0))
        cosmo = self._get_cosmo( H0=H0)
        
        map_val, dlow, dup = self._get_minmax_d(Xi0=Xi0, H0=H0, std_number=std_number, n=n, 
                     Verbose=Verbose, position_val=position_val)
        
        z1 = z(dlow, cosmo, Xi0, n, H0)
        z2 = z(dup, cosmo, Xi0, n, H0)
        if Verbose:
            print('lower limit to search: d_L = %s Mpc, z=%s' %(dlow,z1))
            print('upper limit to search:d_L = %s Mpc, z=%s' %(dup, z2))

        
        minmax_z = max(min(z1,z2), 0), max(z1,z2) 
    
        return minmax_z
    
    def z_range(self, Xi0max=3, Xi0min=0.2, H0max=None, H0min=None, n=1.91, Verbose=False, **params):  
        if Verbose:
            print('\n Computing range in redshift for all priors...')
        
        z_1 = self._get_minmaxz(Xi0max, H0max, Verbose=Verbose,n=n, **params)
        z_2 = self._get_minmaxz(Xi0min, H0min, Verbose=Verbose, n=n,**params)
         
        return (min(z_1+ z_2), max(z_1+z_2))
    
 
 
    def get_CI_from_posterior_sample(self, low_prob=0.01, up_prob=0.99,
                                     data_root='/Users/Michi/Dropbox/Local/Physics_projects/statistical_method_schutz_data_local/GWs/posteriors/',
                                     prior_key='IMRPhenomPv2NRT_lowSpin_prior', post_key='IMRPhenomPv2NRT_lowSpin_posterior'):
        
        import h5py
        
        BBH_file = data_root+self.event_name+'_GWTC-1.hdf5'
        # ---- Posterior samples
        print('-- Posterior')    
        BBH = h5py.File(BBH_file, 'r')
        print('   This file contains datasets: ', BBH.keys())

        med = np.median(BBH[post_key]['luminosity_distance_Mpc'])
        low_lim = np.quantile(BBH[post_key]['luminosity_distance_Mpc'], low_prob)
        low = med-low_lim
        up_lim = np.quantile(BBH[post_key]['luminosity_distance_Mpc'], up_prob)
        up = up_lim-med
        
        return (med, up, low)
    
 
 # --------------------------------------   
 
 #  VISUALIZATION TOOLS
    
 # --------------------------------------

    
def analyse_skymap(data_root, skymap=None, BBH_name=None, 
                   post_sample=True, 
                   prior_key='prior', post_key='Overall_posterior'):

    """
    BBH_name: string with GW event name, e.g. GW170817
    
    Reads skymap and posterior samples of the event with name BBH_name. 
    Computes approximate posterior for D_L from skymap
    Plots the two and prints some statistics to compare
    """
    
    # ---- 3D skymap
    print('-- Skymap')
    if skymap is None:
        fname = data_root+BBH_name+'_LALInference.fits'
        print('Reading file '+fname)
        skymap = skymap_3D(fname, nest=False)
    
    mean_quoted = np.float(dict(skymap.head)['DISTMEAN'])
    std_quoted = np.float(dict(skymap.head)['DISTSTD'])
    
    arr_name = BBH_name+'_dL_post'
    fname = data_root+arr_name
    to_open = fname+'.npy'
    is_there = False
    if os.path.exists(to_open):
        is_there = True
    if not is_there:
        print('    Computing posterior on d_L...' )
        r_list = np.linspace(max(1e-4, mean_quoted-5*std_quoted), mean_quoted+5*std_quoted, 200)
        pr = np.array([skymap.p_r(ri) for ri in r_list])
        print('    Done\n' )
        rpr = np.stack([r_list, pr],axis=1)
        np.save(fname, rpr)
    else:
        loaded = np.load(to_open)
        r_list, pr = loaded[:, 0], loaded[:, 1]
    
    r, pr_h = highres(r_list, pr)
    probs_sm = cumtrapz(pr_h, r,  initial=0)
    med_sm, up_sm, low_sm = find_median_estimate(probs_sm, r, up_p=0.95, low_p=0.05, digits=10)
    mean_sm, std_sm = find_mean_estimate(r, pr_h, digits=15)
    print('Median, 5 lower limit, and 95 upper limit: %s +%s -%s \n' %(int(np.round(med_sm,0)), int(np.round(up_sm,0)), int(np.round(low_sm,0))))
    
    print('My (Mean, std): (%f, %f)' %(mean_sm, std_sm))
    print('Quoted (Mean, std): (%f, %f)\n\n' %(mean_quoted, std_quoted))
    
    if post_sample:
        import h5py
        
        BBH_file = data_root+BBH_name+'_GWTC-1.hdf5'
        # ---- Posterior samples
        print('-- Posterior')    
        BBH = h5py.File(BBH_file, 'r')
        print('   This file contains datasets: ', BBH.keys())

        med = np.median(BBH[post_key]['luminosity_distance_Mpc'])
        low_lim = np.quantile(BBH[post_key]['luminosity_distance_Mpc'], 0.05)
        low = med-low_lim
        up_lim = np.quantile(BBH[post_key]['luminosity_distance_Mpc'], 0.95)
        up = up_lim-med

        print('\n   Median, 5 lower limit, and 95 upper limit: %s + %s -%s \n' %(int(np.round(med,0)), int(np.round(up,0)), int(np.round(low,0))))
        
        plt.hist(BBH[post_key]['luminosity_distance_Mpc'], bins = 100, label='Posterior samples', alpha=0.8, density=True)
        #plt.hist(BNS[post_key_1]['luminosity_distance_Mpc'], bins = 100, label='low spin prior', alpha=0.8, density=True)
    plt.plot(r, pr_h);
    plt.xlabel(r'$D_L (Mpc)$');
    plt.ylabel('Probability Density Function');
    plt.legend();
    plt.show();
    
    return skymap, r_list, pr, (med_sm, up_sm, low_sm), (med, up, low)
 
    

 
