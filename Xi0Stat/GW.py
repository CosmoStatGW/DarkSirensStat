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
from globals import *
from scipy.special import erfinv
from astropy.cosmology import Planck15
from scipy.integrate import quad
from ligo.skymap.io import fits
from os import listdir
from os.path import isfile, join




class Skymap3D(object):
    

    def __init__(self, fname, priorlimits, level=0.99, 
                 nest=False, verbose=False, std_number=None,
                 zLimSelection='skymap'):
        
        self.nest=nest
        self.verbose = verbose
        self.priorlimits = priorlimits
        self.level=level
        if zLimSelection not in ('skymap, header'):
            raise ValueError('Please set a valid option for zLimSelection. Valid entries are: skymap, header')
        self.zLimSelection=zLimSelection
        if std_number is None:
            self.std_number = np.sqrt(2)*erfinv(level)
        else:
            self.std_number=std_number
        
        
        self.read_map(fname) # sets p, mu, sigma, norm
        
        
        self.npix = len(self.p_posterior)
        self.nside = hp.npix2nside(self.npix)
        self.pixarea = hp.nside2pixarea(self.nside, degrees=False) # pixel area in square radians
        #self.p_posterior = smap[0]
        #self.head = header
        #self.mu   = smap[1]
        #self.sigma   = smap[2]
        #self.posteriorNorm   = smap[3]
        self.all_pixels = np.arange(self.npix)
        self.metadata = self._get_metadata()
      
        # the likelihood *does* still contain the posteriorNorm things, or in other words, the posterior p's are not the likelihood "pixel probabilities"
        # we normalize the likelihood to get a pdf for the measure (dOmega d dLGW)
        self.p_likelihood = self.p_posterior*self.posteriorNorm
        # the normalization is a bit subtle. We want to normalize the likelihood in the same way as in the sampling-based evaluation, where we sample the posterior, obtained by combining the likelihood with dLGW^2.
        # this means that the likelihood should be normalized to give the normalized posterior after multiplying by dLGW^2
        # This is the case using the following. The posteriorNorm disappears because
        # for each pixel it drops after doing the ddLGW integral.
        # The angular integral gives sum pixarea * p_posterior_i which needs to be 1.
        self.p_likelihood /= (np.sum(self.p_posterior)*self.pixarea)
        self.set_credible_region()
    
    
    def read_map(self, fname):
        
        if 'O2' in fname.split('/'):
            return self._read_O2(fname)
        else: 
            return self._read_O3(fname)
    
    
    def _read_O3(self, fname, convert_nested=True):
        
        skymap, metadata = fits.read_sky_map(fname, nest=None, distances=True) #Read the skymap
        if len(fname.split('/')[-1].split('_')) == 2:    #Same as before since certain names contain a "_"
            self.event_name = fname.split('/')[-1].split('_')[0]
        elif len(fname.split('/')[-1].split('_')) == 3:
            self.event_name = fname.split('/')[-1].split('_')[0]+'_'+fname.split('/')[-1].split('_')[1]
        else:
            raise ValueError('Could not set event name. Got fname= %s'%fname)
        if self.verbose:
                print('\nEvent: %s' %self.event_name)
        if (convert_nested) & (metadata['nest']): #If one wants RING ordering (the one of O2 data afaik) just has to set "convert_nested" to True
            self.p_posterior = hp.reorder(skymap[0],n2r=True)
            self.mu = hp.reorder(skymap[1],n2r=True)
            self.sigma = hp.reorder(skymap[2],n2r=True)
            self.posteriorNorm = hp.reorder(skymap[3],n2r=True)
        else:
            self.p_posterior= skymap[0]
            self.mu= skymap[1]
            self.sigma = skymap[2]
            self.posteriorNorm= skymap[3]        
        
        self.head = None
    
    
    def _read_O2(self, fname):
        
        try:
            
            smap, header = hp.read_map(fname, field=range(4),
                                       h=True, nest=self.nest, verbose=False)
                 
        except IndexError:
            print('No parameters for 3D gaussian likelihood')
            smap = hp.read_map(fname, nest=nest, verbose=False)
            header=None
        
        try:
            self.event_name=dict(header)['OBJECT']
            if self.verbose:
                print('\nEvent: %s' %self.event_name)
        except KeyError:
            ename = fname.split('/')[-1].split('.')[0].split('_')[0]
            print('No event name in header for this event. Using name provided with filename %s ' %ename)
            self.event_name = ename
        
        self.p_posterior=smap[0]
        self.mu=smap[1]
        self.sigma=smap[2]
        self.posteriorNorm=smap[3]
        self.head = header
        
        
        
    
    
    def set_credible_region(self, level=None):
        if level is None:
            level=self.level    
        px = self.get_credible_region_pixels(level=level)
        pxmask = np.isfinite(self.mu[px]) & (self.mu[px] >= 0)
        self.selected_pixels = px[pxmask]
        self.p_posterior_selected = np.zeros(self.npix)
        self.p_posterior_selected[self.selected_pixels] = self.p_posterior[self.selected_pixels]
        self.p_likelihood_selected = self.p_posterior_selected*self.posteriorNorm
        self.p_likelihood_selected /= (np.sum(self.p_posterior_selected)*self.pixarea)        
        
        self.compute_z_lims()
        
        self.area , self.area_rad,  self.vol  = self._get_credible_region_info()
            
        if self.verbose:
            print('Credible region set to %s %%' %(self.level*100))
            print('Number of std in dL: %s' %self.std_number)
            # Print size of the credible region
            vol="{:.2e}".format(self.vol)
            print('%s credible region for %s: area=%s deg^2 (%s rad^2), volume= %s Mpc^3 (with H0=%s)' %(self.level, self.event_name, np.round(self.area), np.round(self.area_rad, 3), vol, cosmoglob.H0))
        # further remove bad pixels where no skymap is available
        
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
    
    
    def find_event_coords(self, polarCoords=False):
        if not polarCoords:
            return self.find_ra_dec(np.argmax(self.p_posterior))
        else:
            return self.find_theta_phi(np.argmax(self.p_posterior))
    
    def dp_dr_cond(self, r, theta, phi):
        '''
        conditioned probability 
        p(r|Omega)
        
        input:  r  - GW lum distance in Mpc
                ra, dec
        output: eq. 2.19 , cfr 1 of Singer et al 2016
        '''
        pix = self.find_pix(theta, phi)
        return (r**2)*self.posteriorNorm[pix]*scipy.stats.norm.pdf(x=r, loc=self.mu[pix], scale=self.sigma[pix])

    
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
        return cond_p*self.p_posterior_selected[pix] #/self.pixarea
    
    
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
        #myclip_a=0
        #myclip_b=np.infty
        #a, b = (myclip_a - self.mu[pix]) / self.sigma[pix], (myclip_b - self.mu[pix]) / self.sigma[pix]
        #return  self.p_likelihood_selected[pix]*scipy.stats.truncnorm(a=a, b=b, loc=self.mu[pix], scale=self.sigma[pix]).pdf(r)
        
        return self.p_likelihood_selected[pix]*trunc_gaussian_pdf(x=r, mu=self.mu[pix], sigma=self.sigma[pix], lower=0 )
        #scipy.stats.norm.pdf(x=r, loc=self.mu[pix], scale=self.sigma[pix])
    
    
    def sample_posterior(self, nSamples):
        # sample pixels
                 
        def discretesample(nSamples, pdf):
            cdf = np.cumsum(pdf)
            cdf = cdf / cdf[-1]
            return np.searchsorted(cdf, np.random.uniform(size=nSamples))
            
        # norm goes away sampling r^2 as well below, only prob remains to give pixel probability
        
        pixSampled = discretesample(nSamples, self.p_posterior_selected)


        mu = self.mu[pixSampled]
        sig = self.sigma[pixSampled]

        # sample distances. note mu is not the peak location of the *posterior* with r^2. be generous with sigma...
        res = 1000
        lower = mu - 3.5*sig
        np.clip(lower, a_min=0, a_max=None, out=lower)
        upper = mu + 3.5*sig
        grids = np.linspace(lower, upper, res).T
        mu = mu[:, np.newaxis]
        sig = sig[:, np.newaxis]

        pdfs = mu**2*np.exp(-(mu - grids)**2/(2*sig**2))
        # not necessary pdfs /= np.sqrt(2*np.pi)*sig

        rSampled = np.zeros(nSamples)
        for i in np.arange(nSamples):
            idx = np.min((discretesample(1, pdfs[i, :]), res-1))
            rSampled[i] = grids[i, idx]
            
        # sample distances
#        rSampled = sample_trunc_gaussian(self.mu[pix], self.sigma[pix], lower=0, size=1)

        theta, phi = self.find_theta_phi(pixSampled)
        return theta, phi, rSampled
      
    def p_r(self, r):  
        '''
        Posterior on lum. distance p(r|data) 
        marginalized over Omega
        To be compared with posterior chains 
        '''
        return sum(self.p_posterior_selected*self.posteriorNorm*scipy.stats.norm(loc=self.mu, scale=self.sigma).pdf(r) )*r**2
    
    def p_om(self, theta, phi):
        '''
        p(Omega)
        '''
        return self.p_posterior_selected[self.find_pix(theta, phi)]
    
#    def area_p(self, pp=0.9):
#        ''' Area of pp% credible region '''
#        i = np.flipud(np.argsort(self.p_posterior))
#        sorted_credible_levels = np.cumsum(self.p_posterior[i])
#        credible_levels = np.empty_like(sorted_credible_levels)
#        credible_levels[i] = sorted_credible_levels
#        #from ligo.skymap.postprocess import find_greedy_credible_levels
#        #credible_levels = find_greedy_credible_levels(self.p)
#
#        return np.sum(credible_levels <= pp) * hp.nside2pixarea(self.nside, degrees=True)
    
    def area(self, level=None):
        ''' Area of level% credible region, in square degrees.
            If level is not specified, uses current selection '''
            
        if level==None:
            return self.selected_pixels.size*self.pixarea*(180/np.pi)**2
        else:
            return self.get_credible_region_pixels(level=level).size*self.pixarea*(180/np.pi)**2
            
        
    
    def _get_credible_region_pth(self, level=None):
        '''
        Finds value minskypdf of rho_i that bouds the x% credible region , with x=level
        Then to select pixels in that region: self.all_pixels[self.p_posterior>minskypdf]
        '''
        if level is None:
            level=self.level
        prob_sorted = np.sort(self.p_posterior)[::-1]
        prob_sorted_cum = np.cumsum(prob_sorted)
        # find index of array which bounds the self.area confidence interval
        idx = np.searchsorted(prob_sorted_cum, level)
        minskypdf = prob_sorted[idx] #*skymap.npix
        
        #self.p[self.p]  >= minskypdf       
        return minskypdf
    
    def get_credible_region_pixels(self, level=None):
        if level is None:
            level=self.level
        return self.all_pixels[self.p_posterior>self._get_credible_region_pth(level=level)]
    
    
#    def likelihood_in_credible_region(self, r, level=0.99, verbose=False):
#        '''
#        Returns likelihood for all the pixels in the x% credible region at distance r
#        x=level
#        '''
#        cArea_idxs = self.get_credible_region_pixels(level=level)
#        LL = self.likelihood_px(r, cArea_idxs)
#        if verbose:
#            print('Max GW likelihood at dL=%s Mpc : %s' %(r,LL.max()))
#            print('Pix of max GW likelihood = %s' %cArea_idxs[LL.argmax()])
#            print('RA, dec of max GW likelihood at dL=%s Mpc: %s' %(r,self.find_ra_dec(cArea_idxs[LL.argmax()])))
#        return LL
#
#
    def d_max(self, SNR_ref=8):
        '''
        Max GW luminosity distance at which the evend could be seen, 
        assuming its SNR and a threshold SNR_ref:
        d_max = d_obs*SNR/SNR_ref
    
        '''
        
        d_obs, _, _, _ = self.find_r_loc()
        
        try:
            #d_obs = self.metadata['luminosity_distance'].values[0]
            
            SNR = self.metadata['network_matched_filter_snr'].values[0]
            return d_obs*SNR/SNR_ref
            #print('using d_obs and SNR from metadata')
            
        except IndexError:
            print('SNR for this event not available! Scaling event distance by 1.5...')
            return 1.5*d_obs
          
      
#
#    def compute_z_lims(self, H0max=220, H0min=20, Xi0max=3, Xi0min=0.2, n=1.91, verbose=False):
#        '''
#        Computes the possible range in redshift for the event given the prior range for H0 and XI0
#        '''
#
#        if verbose:
#            print('Computing range in redshift for all priors...')
#        grid=[]
#        for H0i in [H0min, H0max]:
#            for Xi0i in [Xi0min, Xi0max]:
#                grid.append([H0i, Xi0i])
#        zs = np.array([self._get_minmax_z(*vals,n=n, verbose=verbose) for vals in grid ])
#
#        self.zmin = zs.min()
#        self.zmax = zs.max()
#        return self.zmin, self.zmax
#

    def compute_z_lims(self, std_number=None, n=1.91):
        '''
        Computes and stores z range of events given H0 and Xi0 ranges.
        Based on actual skymap shape in the previously selected credible region, not on metadata
        '''
        if std_number is None:
            std_number=self.std_number
        if self.verbose:
            print('Computing range in redshift for parameter range H0=[{}, {}], Xi0=[{}, {}]...'.format(self.priorlimits.H0min, self.priorlimits.H0max, self.priorlimits.Xi0min, self.priorlimits.Xi0max))
        
        self.dL, self.dmin, self.dmax, self.sigmadL = self.find_r_loc(std_number=std_number)
        
        self.zmin = z_from_dLGW(self.dmin, self.priorlimits.H0min, self.priorlimits.Xi0max, n=n)
        self.zmax = z_from_dLGW(self.dmax, self.priorlimits.H0max, self.priorlimits.Xi0min, n=n)
        self.zEv = z_from_dLGW(self.dL, H0GLOB, Xi0Glob, n=n)

        return self.zmin, self.zmax, self.dmin, self.dmax
        
    def get_z_lims(self):
        '''
        Returns z range of events as computed for given H0 and Xi0 ranges.
        Based on actual skymap shape in the selected credible region, not metadata
        '''
        return self.zmin, self.zmax
    
    
    def find_r_loc(self, std_number=None, verbose=None):
        if verbose is None:
            verbose=self.verbose
        if self.zLimSelection=='skymap':
            if verbose:
                print('DL range computed from skymap')
            return self._find_r_loc(std_number=std_number)
        else:
            if verbose:
                print('DL range computed from header')
            return self._metadata_r_lims(std_number=std_number)
        
    
    def _find_r_loc(self, std_number=None):
        '''
        Returns mean GW lum. distance, lower and upper limits of distance, and the mean sigma.
        Based on actual skymap shape in the selected credible region, not metadata.
        '''
        if std_number is None:
            std_number=self.std_number
        mu = self.mu[self.selected_pixels]
        sigma = self.sigma[self.selected_pixels]
        p = self.p_likelihood_selected[self.selected_pixels]
        
        meanmu = np.average(mu, weights = p)
        meansig = np.average(sigma, weights = p)
        lower = max(np.average(mu-std_number*sigma, weights = p), 0)
        upper = np.average(mu+std_number*sigma, weights = p)
        if self.verbose:
            print('Position: %s +%s %s'%(meanmu, upper, lower))
        
        return meanmu, lower, upper, meansig
        
    
    def _metadata_r_lims(self, std_number=None):
        '''
        "Official" limits based on metadata - independent of selected credible region
        '''
        if std_number is None:
            std_number=self.std_number
        mean = np.float(dict(self.head)['DISTMEAN'])
        std = np.float(dict(self.head)['DISTSTD'])
        
        lower = max(mean-std_number*std,0)
        upper=mean+std_number*std
        if self.verbose:
            print('Position: %s +%s %s'%(mean, upper, lower))
        meanmu = map_val
        meansig = up_lim
        
        return mean, lower, upper, std #map_val, up_lim, low_lim
        
#    
    def _get_credible_region_info(self):
        area_deg = self.area()
        area_rad = area_deg/((180/np.pi)**2)
        #_, dmin, dmax, _ = self.find_r_loc()
        zmin = z_from_dLGW(self.dmin, cosmoglob.H0.value, 1,n=1.91)
        zmax = z_from_dLGW(self.dmax, cosmoglob.H0.value, 1, n=1.91)
        com_vol = quad(lambda x: cosmoglob.differential_comoving_volume(x).value, zmin, zmax )[0]
        vol=area_rad*com_vol
        return area_deg, area_rad, vol
    
    #def _get_minmax_z(self, H0, Xi0, n=1.91, std_number=3):
#        '''
#        Upper and lower limit in redshift to search for given H0 or Xi0
#        Based on selected credible region.
#        '''
#        _, dlow, dup, _ = find_r_loc(std_number=std_number)
#
#        z1 = z_from_dLGW(dlow, H0, Xi0, n=n)
#        z2 = z_from_dLGW(dup,  H0, Xi0, n=n)
#
#        if self.verbose:
#            print('H0, Xi0: %s, %s' %(H0, Xi0))
#            print('lower limit to search: d_L = %s Mpc, z=%s' %(dlow,z1))
#            print('upper limit to search:d_L = %s Mpc, z=%s' %(dup, z2))
#
#        return z1, z2
        #minmax_z = max(min(z1,z2), 0), max(z1,z2)
    
        #return minmax_z
            
    def adap_z_grid(self, H0, Xi0, n, zR=zRglob, eps=1e-03):
        meanmu, lower, upper, meansig = self.dmin, self.dmax, self.dL, self.sigmadL #self.find_r_loc(std_number=5)
    
        zLow =  max(z_from_dLGW(lower, H0, Xi0, n), 1e-7)
        zUp = z_from_dLGW(upper, H0, Xi0, n)
        #print(zLow, zUp)
        z_grid=np.concatenate([np.log10(np.logspace(0, zLow, 50)), np.linspace(zLow+eps, zUp, 100),  np.linspace(zUp+eps, zUp+0.1, 20), np.linspace(zUp+0.1+eps, zR, 50)])
    
        return z_grid
  





def get_all_events(priorlimits, loc='data/GW/O2/', 
                   wf_model_name='PublicationSamples',
                   subset=False, 
                   subset_names=['GW170817',],
                   verbose=False, **kwargs
               ):
    
    if 'O2' in loc.split('/'):
        run='O2'
        sm_files = [f for f in listdir(join(dirName,loc)) if ((isfile(join(dirName, loc, f))) and (f!='.DS_Store') and  ('skymap' in f) )]
        ev_names = [fname.split('_')[0]  for fname in sm_files]
    elif 'O3' in loc.split('/'):
        run='O3'
        sm_files = [f for f in listdir(join(dirName,loc)) if ((isfile(join(dirName, loc, f))) and (f!='.DS_Store') and (wf_model_name+'.fits' in f.split('_')) )]
        ev_names = [] #Initialize array
        #Event names could display the time separated by a "_", for this reason the next two if are necessary
        #Maybe this is not the best way, but it works and is fast
        for fname in sm_files:
            if len(fname.split('_')) == 2: 
                ev_names.append(fname.split('_')[0])
            elif len(fname.split('_')) == 3:
                ev_names.append(fname.split('_')[0]+'_'+fname.split('_')[1])
        #ev_names = [e+'_'+wf_model_name for e in ev_names]
    else:
        raise ValueError('O2 or O3 data expected.')
    
    if verbose:
        print('GW observing run: %s' %run)
        print('Found %s skymaps: %s' %(len(sm_files), str(ev_names)))
    
    if sum([fname.endswith('.gz') for fname in sm_files])==len(sm_files):
        compressed=True
    else:
        compressed=False
        
    if subset:
        ev_names = [e for e in ev_names if e in subset_names]
        if run=='O2':
            if compressed:
                sm_files = [e+'_skymap.fits.gz' for e in ev_names]
            else:
                sm_files = [e+'_skymap.fits' for e in ev_names]
        else:
            if compressed:
                sm_files = [e+'_'+wf_model_name+'.fits.gz' for e in ev_names]
            else:
                sm_files = [e+'_'+wf_model_name+'.fits' for e in ev_names]
    if verbose:
        #print('GW events:')
        #print(ev_names)
        print('Reading skymaps: %s...' %str(ev_names))
        all_events = {fname.split('_')[0]: Skymap3D(os.path.join(dirName,loc,fname), priorlimits=priorlimits, nest=False, verbose=verbose, **kwargs) for fname in sm_files}

    return all_events




def get_all_events_1(priorlimits, loc='data/GW/O2/', 
                   wf_model_name='PublicationSamples',
                   subset=False, 
                   subset_names=['GW170817',],
                   verbose=False, **kwargs
               ):
    '''
    Returns dictionary with all skymaps in the folder loc.
    If subset=True, gives skymaps only for the event specified by subset_names
    
    '''

    
    sm_files, ev_names, compressed = get_file_list(loc, verbose=verbose)
    
    if subset:
        ev_names = [e for e in ev_names if e in subset_names]
        if compressed:
            sm_files = [e+'.fits.gz' for e in ev_names]
        else:
            sm_files = [e+'.fits' for e in ev_names]
    if verbose:
        #print('GW events:')
        #print(ev_names)
        print('Reading skymaps: %s...' %str(sm_files))
    all_events = {fname.split('_')[0]: Skymap3D(os.path.join(dirName,loc,fname), priorlimits=priorlimits, nest=False, verbose=verbose, **kwargs) for fname in sm_files}
    return all_events
    
