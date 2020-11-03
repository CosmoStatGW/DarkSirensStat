#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:53:55 2020

@author: Michi
"""

from config_Andreas import *
import os
import time
import sys
import shutil


from globals import *
from GW import get_all_events
from GLADE import GLADE
from GWENS import GWENS
#from SYNTH import SYNTH
from completeness import *
from galCat import GalCompleted
from GWgal import GWgal
from betaHom import BetaHom
from betaFit import BetaFit
from betaMC import BetaMC

from plottingTools import plot_completeness, plot_post



def completeness_case(completeness, band, Lcut, path=None):
    
    
    if band is None:
        goal=Nbar
    elif band=='B':
        goal=get_SchNorm(phistar=phiBstar07, Lstar=LBstar07, alpha=alphaB07, Lcut=Lcut)
    elif band=='K':
        goal=get_SchNorm(phistar=phiKstar07, Lstar=LKstar07, alpha=alphaK07, Lcut=Lcut)
    else:
        raise ValueError('Enter a valid value for band. Valid options are: B, K, None')
        
    if completeness=='load':
        compl = LoadCompleteness( completeness_path, goal, interpolateOmega)
    elif completeness=='pix':
        compl = SuperpixelCompleteness(goal, angularRes, zRes, interpolateOmega)
    elif completeness=='mask':
        compl = MaskCompleteness(goal, zRes, nMasks=nMasks)
    elif completeness=='skip':
        compl=SkipCompleteness()
    else:
        raise ValueError('Enter a valid value for completeness. Valid options are: load, pix, mask, skip')
    
    return compl


def beta_case(which_beta, allGW, lims, H0grid, Xi0grid):
    # 'fit', 'MC', 'hom', 'cat'
    
    if which_beta in ('fit', 'MC'):
        if which_beta=='fit':
            Beta = BetaFit(zR)
        elif which_beta=='MC':
            Beta = BetaMC(lims, nSamples=nMCSamplesBeta, observingRun = observingRun, SNRthresh = SNRthresh)
        beta = Beta.get_beta(H0grid, Xi0grid)
        betas = {event:beta for event in allGW}
    elif which_beta in ('hom', 'cat'):
        if which_beta=='hom':
            betas = {event: BetaHom( allGW[event].d_max(), zR).get_beta(H0grid, Xi0grid) for event in allGW.keys()}
            #Beta = BetaHom( dMax, zR)
        elif which_beta=='cat':
            raise NotImplementedError('Beta from catalogue is not supported for the moment. ')
    else:
        raise ValueError('Enter a valid value for which_beta. Valid options are: fit, MC, hom, cat')
    return betas
    


def get_norm_posterior(lik_inhom,lik_hom, beta, grid):
    
    tot_post = (lik_inhom+lik_hom)/beta
    norm = np.trapz( tot_post, grid)
    
    post = tot_post/norm
    return post, lik_inhom/beta/norm, lik_hom/beta/norm



def main():
    
    in_time=time.time()
    
    # St out path and create out directory
    out_path=os.path.join(dirName, 'results', fout)
    if not os.path.exists(out_path):
        print('Creating directory %s' %out_path)
        os.makedirs(out_path)
    else:
       print('Using directory %s for output' %out_path)
    
    logfile = os.path.join(out_path, 'logfile.txt') #out_path+'logfile.txt'
    myLog = Logger(logfile)
    sys.stdout = myLog
    
    shutil.copy('config.py', os.path.join(out_path, 'config_original.py'))
    
    
    
    ###### 
    # GW data
    ######
    
    print('\n-----  LOADING GW DATA ....')
    
    data_loc = os.path.join('data','GW', observingRun )
    
    # Set prior range
    lims = PriorLimits() 
    lims.Xi0min=Xi0min
    lims.Xi0max=Xi0max
    lims.H0min=H0min
    lims.H0max=H0max
    
    if subset_names==None:
        subset=False
    else: subset=True
    
    allGW = get_all_events(loc=data_loc,
                    priorlimits=lims ,
                    subset=subset, subset_names=subset_names, 
                    verbose=True, level = level, std_number=std_number, )#compressed=is_compressed)
    
    
    
    ###### 
    # Completeness
    ######
    if completeness=='load':
        compl =  completeness_case(completeness, band, Lcut, completeness_path)
    else:
        compl =  completeness_case(completeness, band, Lcut)
    
    
    ###### 
    # Galaxy catalogues
    ######
    
    print('\n-----  LOADING GALAXY CATALOGUES DATA ....')
    
    gals = GalCompleted(completionType=completionType)
    
    
    if catalogue in ('GLADE', 'MINIGLADE'):
    
        if fastglade:
            cat = GLADE('GLADE', compl, useDirac=False, finalData='posteriorglade.csv', verbose=True, band=band, Lcut=Lcut)
        else:
            cat = GLADE('GLADE', compl, useDirac, band=band, Lcut=Lcut, verbose=True,
              computePosterior=computePosterior)
        gals.add_cat(cat)
        
    elif catalogue == 'GWENS':
        cat = GWENS('GWENS', compl, useDirac, verbose=True)
    else:
        raise NotImplementedError('Galaxy catalogues other than GLADE or GWENS are not supported for the moment. ')
    if plot_comp:
        plot_completeness(out_path, allGW, cat)
    
    ###### 
    # GWgal
    ######
    myGWgal = GWgal(gals, allGW, MC=MChom, nHomSamples=nHomSamples, verbose=True, galRedshiftErrors=galRedshiftErrors, zR=zR)
    myGWgal._select_events(PavMin=PavMin, PEvMin=PEvMin)
    
    if plot_comp:
        plot_completeness(out_path, myGWgal.selectedGWevents, cat)
    
    ###### 
    # Grids
    ######
    if goalParam=='H0':
        H0grid=np.linspace(lims.H0min, lims.H0max, nPointsPosterior)
        Xi0grid=Xi0Glob
        grid=H0grid
    elif goalParam=='Xi0':
        H0grid=H0GLOB
        Xi0grid=np.linspace(lims.Xi0min, lims.Xi0max, nPointsPosterior)
        grid=Xi0grid
    np.savetxt(os.path.join(out_path, goalParam+'_grid.txt'), grid)
    
    
    ###### 
    # Beta
    ######
    if do_inference:
        print('\n-----  COMPUTING BETAS ....')
        betas = beta_case(which_beta, myGWgal.selectedGWevents, lims, H0grid, Xi0grid)
        for event in myGWgal.selectedGWevents:
            betaPath =os.path.join(out_path, event+'_beta'+goalParam+'.txt')
            np.savetxt(betaPath, betas[event])
        print('Done.') 

    
        ###### 
        # Likelihood
        ######
        print('\n-----  COMPUTING LIKELIHOOD ....')
        liks = myGWgal.get_lik(H0s=H0grid, Xi0s=Xi0grid, n=nGlob)
        for event in myGWgal.selectedGWevents:
            liksPathhom =os.path.join(out_path, event+'_lik_compl'+goalParam+'.txt')
            liksPathinhom =os.path.join(out_path, event+'_lik_cat'+goalParam+'.txt')
            np.savetxt(liksPathhom, liks[event][1])
            np.savetxt(liksPathinhom, liks[event][0])
        print('Done.')
    
    
        print('\n-----  COMPUTING POSTERIOR ....')
    
        post, post_cat, post_compl = {},  {},  {}
        for event in myGWgal.selectedGWevents.keys():
            post[event], post_cat[event], post_compl[event] = get_norm_posterior(liks[event][0],liks[event][1], betas[event], grid)
    
        if goalParam=='H0':
            myMin, myMax= lims.H0min, lims.H0max
        else:  myMin, myMax= lims.Xi0min, lims.Xi0max
    
        post, post_cat, post_compl = plot_post(out_path, grid, post, post_cat, post_compl, myGWgal.selectedGWevents.keys(),
                                              band,Lcut,zR,
                                              myMin=myMin, myMax=myMax, 
                                              varname=goalParam,)
    
        for event in myGWgal.selectedGWevents:
            postPathhom =os.path.join(out_path, event+'_post_compl'+goalParam+'.txt')
            postPathinhom =os.path.join(out_path, event+'_post_cat'+goalParam+'.txt')
            postPathtot=os.path.join(out_path, event+'_post'+goalParam+'.txt')
            np.savetxt(postPathhom, post_compl[event])
            np.savetxt(postPathinhom, post_cat[event])
            np.savetxt(postPathtot, post[event])
    else:
        myGWgal.select_gals()
        
    myGWgal._get_summary()
    #summary = myGWgal.summary()
    myGWgal.summary.to_csv(os.path.join(out_path, 'summary.csv') )
    
    print('\nDone in %.2fs' %(time.time() - in_time))
    
    sys.stdout = sys.__stdout__
    myLog.close()    
    
    





if __name__=='__main__':
    main()
