#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 16:04:06 2020

@author: Michi
"""

from config_counterpart import *
import os
import time
import sys
import shutil
import argparse



from globals import *
from betaHom import BetaHom
from betaFit import BetaFit
from betaMC import BetaMC
from betaCat import BetaCat
from eventSelector import *

from main import beta_case, get_norm_posterior
from plottingTools import  *

#Packages to read posteriors
import pesummary
from pesummary.io import read
import h5py

#Packages for KDE
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut

from scipy.stats import norm



def fit_spline(ra_spls, dec_spls, dL_spls, dL_prior_spls, 
                   ra_c, dec_c, z_c, 
                   kde_bandwidth, cone_area=0.0045):
    phi0, theta0 = ra_c*(np.pi/180), np.pi/2-dec_c*(np.pi/180)
    psi = 2*np.arccos(1 - cone_area/(2*np.pi)) #opening angle cone
    
    mask = haversine(ra_spls, np.pi/2-(dec_spls), phi0, theta0) < psi
    data_dL=dL_spls[mask]
    
    kde_post = KernelDensity(bandwidth=kde_bandwidth, kernel='gaussian')
    if dL_prior_spls is not None:
        kde_prior = KernelDensity(bandwidth=kde_bandwidth, kernel='gaussian')
        kde_prior.fit(dL_prior_spls[:, None])
        logprob = kde_prior.score_samples(data_dL[:, None])
        weights = np.exp(-logprob)
    else:
        weights = data_dL**(-2)      
    kde_post.fit(data_dL[:, None],sample_weight=weights)
    return kde_post


def compute_dL_Lik(grid, kde_post):
    if not np.isscalar(grid):
        logprob = kde_post.score_samples(grid[:, None])
    else:
        logprob = kde_post.score_samples(grid.reshape(1, -1) )  
    return np.exp(logprob)




def read_full_post(file_name,wf_model,file_ext='hdf5',prior_file_name=None):
    
    print('Reading posterior samples from %s...' %file_name)
    
    if 'hdf5' in file_ext:
        data_in = h5py.File(file_name, 'r')
        data_use = data_in[wf_model+'_posterior']
        
        if 'ra' in data_use.dtype.names:
            ra_name_use = 'ra'
        elif 'right_ascension' in data_use.dtype.names:
            ra_name_use = 'right_ascension'
        if 'dec' in data_use.dtype.names:
            dec_name_use = 'dec'
        elif 'declination' in data_use.dtype.names:
            dec_name_use = 'declination'
        if 'luminosity_distance' in data_use.dtype.names:
            dL_name_use = 'luminosity_distance'
            kde_bandwidth=300.
        elif 'luminosity_distance_Mpc' in data_use.dtype.names:
            dL_name_use = 'luminosity_distance_Mpc'
            kde_bandwidth=2.
            
        ra_spls = data_use[ra_name_use]
        dec_spls = data_use[dec_name_use]
        dL_spls = data_use[dL_name_use]
        dL_prior_spls = data_in[wf_model+'_prior'][dL_name_use]
        
    elif 'h5' in file_ext:
        data = read(file_name)
        samples_dict = data.samples_dict
        posterior_samples = samples_dict[wf_model]
        if 'ra' in posterior_samples.keys():
            ra_name_use = 'ra'
        elif 'right_ascension' in posterior_samples.keys():
            ra_name_use = 'right_ascension'
        if 'dec' in posterior_samples.keys():
            dec_name_use = 'dec'
        elif 'declination' in posterior_samples.keys():
            dec_name_use = 'declination'
        if 'luminosity_distance' in posterior_samples.keys():
            dL_name_use = 'luminosity_distance'
            kde_bandwidth=300.
        elif 'luminosity_distance_Mpc' in posterior_samples.keys():
            dL_name_use = 'luminosity_distance_Mpc'
            kde_bandwidth=2.
            
        ra_spls = posterior_samples[ra_name_use]
        dec_spls = posterior_samples[dec_name_use]
        dL_spls = posterior_samples[dL_name_use]
        
        if prior_file_name is not None:
            print('Reading prior samples from %s...' %prior_file_name)
            prior_samples=np.load(prior_file_name)
            dL_prior_spls = prior_samples[dL_name_use]
        else:
            prior_samples = None
    return ra_spls, dec_spls, dL_spls, dL_prior_spls, kde_bandwidth




def main_counterpart():
    
    in_time=time.time()
    

    
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--observingRun", default='O2', type=str, required=True)
    #parser.add_argument("--wf_model", default='', type=str, required=True)
    #FLAGS = parser.parse_args()
    
    #####
    # Out file
    #####
    
    
    # St out path and create out directory
    out_path=os.path.join(dirName, 'results', fout+'_counterpart_'+wf_model)
    if not os.path.exists(out_path):
        print('Creating directory %s' %out_path)
        os.makedirs(out_path)
    else:
       print('Using directory %s for output' %out_path)
       
       
    logfile = os.path.join(out_path, 'logfile.txt') #out_path+'logfile.txt'
    myLog = Logger(logfile)
    sys.stdout = myLog
    
    shutil.copy('config_counterpart.py', os.path.join(out_path, 'config_counterpart_original.py'))
    
    #####
    #####
    
    #######
    
    # Set prior range
    lims = PriorLimits() 
    lims.Xi0min=Xi0min
    lims.Xi0max=Xi0max
    lims.H0min=H0min
    lims.H0max=H0max
    
    ######
    
    dataPath = os.path.join(baseGWPath, observingRun+'_counterpart')
    
    if observingRun is 'O2':
        counterparts = O2_counterparts
    elif observingRun is 'O3':
        counterparts = O3_counterparts
    else:
        raise ValueError('No counterparts outside O2 or O3')
    
    all_liks={}
    all_posts={}
    for counterpart_name in  counterparts.keys():
    
        print('\n----- %s....' %counterpart_name)
        
        if observingRun is 'O2':
            fname =counterpart_name+'_GWTC-1'
            file_ext = '.hdf5'
            prior_file_name=None
            #wf_model = wf_model
            #selec_deg2=0
            file_name=os.path.join(dataPath, fname+file_ext)
        elif observingRun is 'O3':
            fname = counterpart_name
            file_ext = '.h5'
            prior_file_name= os.path.join(dataPath, fname, fname+'_prior.npy')
            #wf_model = FLAGS.wf_model
            #selec_deg2=0.0045
            file_name=os.path.join(dataPath, fname,fname+file_ext)
        
    
        
        ra_spls, dec_spls, dL_spls, dL_prior_spls, kde_bandwidth = read_full_post(file_name, 
                                                                              wf_model, 
                                                                              file_ext=file_ext,
                                                                              prior_file_name=prior_file_name)
        
        ra_c = counterparts[counterpart_name]['RA']
        dec_c = counterparts[counterpart_name]['dec']
        z_c = counterparts[counterpart_name]['z_mean']
        sigma_c = counterparts[counterpart_name]['z_std']
        
    
        print('\nCoordinates of the counterpart: (z, RA, dec) = (%s, %s, %s)' %(z_c, ra_c, dec_c))
        
        ###### 
        # Grids
        ######
        if goalParam=='H0':
            H0grid=np.linspace(lims.H0min, lims.H0max, nPointsPosterior)
            Xi0grid=Xi0Glob
            grid=H0grid
            dLgrid = np.array([dLGW(z_c, H0i, Xi0Glob, nGlob) for H0i in H0grid])
        elif goalParam=='Xi0':
            H0grid=H0GLOB
            Xi0grid=np.linspace(lims.Xi0min, lims.Xi0max, nPointsPosterior)
            grid=Xi0grid
            dLgrid = np.array([dLGW(z_c, H0GLOB, Xi0i, nGlob) for Xi0i in Xi0grid])
        np.savetxt(os.path.join(out_path, goalParam+'_grid.txt'), grid)
        
        
       
    


        ###### 
        # Likelihood
        ######
        print('\n---  COMPUTING LIKELIHOOD....')
        
        kde_post = fit_spline(ra_spls, dec_spls, dL_spls, dL_prior_spls, 
                       ra_c, dec_c, z_c, 
                       kde_bandwidth, cone_area=cone_area)
        
        if sigma_c==0:
            lik = compute_dL_Lik(dLgrid, kde_post)
        else:
            print('Convolving with normal distribution inr redshift with mu=%s, sigma=%s ... ' %(z_c, sigma_c))
            lik=np.empty(grid.shape)
            for i,val in enumerate(grid):
                zgrid = np.linspace(max(0., z_c-5*sigma_c),z_c+5*sigma_c, 100)
                NN = norm.pdf(zgrid, loc=z_c, scale=sigma_c)
                if goalParam=='H0':
                    vals=NN*np.squeeze(np.array([ compute_dL_Lik(dLGW(z, val, Xi0Glob, nGlob), kde_post) for z in zgrid]))
                else:
                    vals=NN*np.squeeze(np.array([ compute_dL_Lik(dLGW(z, H0GLOB, val, nGlob), kde_post) for z in zgrid]))
                lik[i] = np.trapz(vals, zgrid)
        
        likPat =os.path.join(out_path, counterpart_name+'_lik'+goalParam+'.txt')

        np.savetxt(likPat, lik)
        
        all_liks[counterpart_name]=lik
        print('Done.')
        
        
        ############
        # Beta
        ############
        print('\n---  COMPUTING BETA ....')
        
        if which_beta=='MC':
            Beta = BetaMC(lims, SkipSelection(), gals=None, 
                      nSamples=nSamplesBetaMC, 
                      observingRun = observingRun, 
                      SNRthresh = SNRthresh, 
                      properAnisotropy=False, verbose=verbose )
        elif which_beta=='hom':
            Beta=BetaHom(betaHomdMax, zR)
            beta = Beta.get_beta(H0grid, Xi0grid)
        else:
            raise ValueError('Only beta MC or hom are supported.')
        
        
        
        print('Done.')
        ############
        # Posterior
        ############
        print('\n---  COMPUTING POSTERIOR ....')
        post, _ , _ = get_norm_posterior(lik, np.zeros(lik.shape), beta, grid)
        postPath =os.path.join(out_path, counterpart_name+'_post'+goalParam+'.txt')
        np.savetxt(postPath, post)
        all_posts[counterpart_name]=post
    
    
        print('Done.')
    
    
    ############
    # Save plots
    #############
    if goalParam=='H0':
            myMin, myMax= lims.H0min, lims.H0max
    else:  myMin, myMax= lims.Xi0min, lims.Xi0max
    _, _, _ = plot_post(out_path, grid, 
                                           all_posts, 
                                           post_cat=None, 
                                           post_compl=None, 
                                           event_list=list(counterparts.keys()),
                                           myMin=myMin, myMax=myMax, 
                                           varname=goalParam,)
    
    ######
    print('\nDone in %.2fs' %(time.time() - in_time))
    
    sys.stdout = sys.__stdout__
    myLog.close() 


if __name__=='__main__':
    main_counterpart()

    
