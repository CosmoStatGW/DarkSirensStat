#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:53:55 2020

@author: Michi
"""

from config import *
import os
import time
import sys
import shutil


from globals import *
from GW import get_all_events
from GLADE import GLADE
from GWENS import GWENS
from DES import DES
#from SYNTH import SYNTH
from completeness import *
from galCat import GalCompleted
from GWgal import GWgal
from betaHom import BetaHom
from betaFit import BetaFit
from betaMC import BetaMC
from betaCat import BetaCat
from pathlib import Path
from eventSelector import *
import json

from plottingTools import plot_completeness, plot_post




def check_footprint(allGw, observingRun, subset=False, DES=True, GWENS=True):
    
    get_DES = DES
    get_GWENS=GWENS
    
    level=allGw[list(allGw.keys())[0]].level
    
    #if not subset:
    DES_file = Path(os.path.join(dirName, 'data', 'DES', 'DES_footprint_'+observingRun+'_level'+str(level)+'.json'))
    GWENS_file = Path(os.path.join(dirName, 'data', 'GWENS', 'GWENS_footprint_'+observingRun+'_level'+str(level)+'.json'))

    #else:
        #DES_file = Path(os.path.join(dirName, 'data', 'DES', 'DES_footprint_'+observingRun+'-'.join(list(allGw.keys()))+'.json'))
        #GWENS_file = Path(os.path.join(dirName, 'data', 'GWENS', 'GWENS_footprint_'+observingRun+'-'.join(list(allGw.keys()))+'.json'))
    if DES_file.is_file() and DES:
        with open(DES_file) as f:
            print('Reading DES coverage from %s...' %DES_file)
            fp_DES = json.load(f)
        get_DES=False
    if GWENS_file.is_file() and GWENS:
        with open(GWENS_file) as f:
            print('Reading GWENS coverage from %s...' %GWENS_file)
            fp_GWENS = json.load(f)
        get_GWENS=False
        
    if get_DES or get_GWENS:
        #all_fp={eventName: {} for eventName in allGw.keys()}
        fp_DES={}
        fp_GWENS={}
        if get_DES:
            DES_fp_path = os.path.join(dirName, 'data', 'DES', 'y1a1_gold_1.0.2_wide_footprint_4096.fits')
            DES_fp = hp.read_map(DES_fp_path,field=[0],verbose=verbose)
        
        if get_GWENS:
            GWENS_fp_path = os.path.join(dirName, 'data', 'GWENS', 'GWENS.footprint_1024.fits')
            GWENS_fp = hp.read_map(GWENS_fp_path,field=[0],verbose=verbose)
        
        for eventName in allGw.keys():
            sm_pxs = allGw[eventName].get_credible_region_pixels()
            if get_DES:
                DES_px = hpx_downgrade_idx(DES_fp, nside_out=allGw[eventName].nside)
                fp_DES[eventName] = np.isin(sm_pxs, DES_px).sum()/sm_pxs.shape[0]
            if get_GWENS:
                GWENS_px = hpx_downgrade_idx(GWENS_fp, nside_out=allGw[eventName].nside)
                fp_GWENS[eventName] = np.isin(sm_pxs, GWENS_px).sum()/sm_pxs.shape[0]
    if get_DES and not subset:
          with open(DES_file, 'w') as json_file:
              print('Saving DES coverage to %s...' %DES_file)
              json.dump(fp_DES, json_file)
    if get_GWENS and not subset:
          with open(GWENS_file, 'w') as json_file:
              print('Saving GWENS coverage to %s...' %GWENS_file)
              json.dump(fp_GWENS, json_file)
        
    return fp_DES, fp_GWENS
    
    
    



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


def beta_case(which_beta, allGW, lims, H0grid, Xi0grid, eventSelector, gals, massDist, lamb):
    # 'fit', 'MC', 'hom', 'cat'
    
    if which_beta in ('fit', 'MC', 'cat'):
        if which_beta=='fit':
            Beta = BetaFit(zR)
        elif which_beta=='MC':
            galsBeta = None
            if nUseCatalogBetaMC:
                galsBeta = gals
            anisotropy = False
            if nUseCatalogBetaMC or type(eventSelector) is not SkipSelection:
                anisotropy = True
            if 'O3' in observingRun:
                observingRunBeta='O3'
            else:
                observingRunBeta=observingRun 

            Beta = BetaMC(lims, eventSelector, gals=galsBeta, nSamples=nSamplesBetaMC, observingRun = observingRunBeta, SNRthresh = SNRthresh, properAnisotropy=anisotropy, verbose=verbose , massDist=massDist, lamb=lamb)
        elif which_beta=='cat':
            Beta=BetaCat(gals, galRedshiftErrors,  zR, eventSelector )
        beta = Beta.get_beta(H0grid, Xi0grid)
        betas = {event:beta for event in allGW}
    elif which_beta in ('hom', ):
        if which_beta=='hom':
            if betaHomdMax == 'scale':
                betas = {event: BetaHom( allGW[event].d_max(), zR).get_beta(H0grid, Xi0grid) for event in allGW.keys()}
            elif betaHomdMax == 'flat':
                betas = {event: BetaHom( allGW[event].dL, zR).get_beta(H0grid, Xi0grid) for event in allGW.keys()}
            else:
                betas = {event: BetaHom(betaHomdMax, zR).get_beta(H0grid, Xi0grid) for event in allGW.keys()}
            #Beta = BetaHom( dMax, zR)
        #elif which_beta=='cat':
            #betas = {event: BetaCat(gals, galRedshiftErrors,  EventSelector ).get_beta(allGW[event], H0grid, Xi0grid) for event in allGW.keys()}
            #raise NotImplementedError('Beta from catalogue is not supported for the moment. ')
    else:
        raise ValueError('Enter a valid value for which_beta. Valid options are: fit, MC, hom, cat')
    return betas
    



def main():
    
    in_time=time.time()
    
    # Consistency checks
    if which_beta == 'fit' and completionType != 'add':
        pass
        #raise ValueError('Beta from fit is obtained from homogeneous completion. Using non-homogeneous completion gives non-consistent results')
    if which_beta == 'cat' and completionType != 'mult':
        raise ValueError('Beta from catalogue is implemented only for multiplicative completion. Use beta=fit or beta=MC')
    if completnessThreshCentral>0. and ( which_beta == 'fit' or which_beta == 'hom') :
        print('\n!!!! completnessThreshCentral is larger than zero but beta %s is used. This beta does not take into account completeness threshold. You may be fine with this, but be aware that the result could be inconsistent.\n' %which_beta)
    if completnessThreshCentral>0. and which_beta == 'MC':
        print('\n!!!! completnessThreshCentral is larger than zero, be sure that beta MC is implementing the completeness threshold.')
        
    if 'NS' in massDist and eventType!='BNS':
        raise ValueError('Using NS mass distribution for BBHs does not make sense. Use massDist=02 or O3. ')
    
    if band_weight is not None:
        if band_weight!=band:
            raise ValueError('Band used for selection and band used for luminosity weighting should be the same ! ')
   # Andreas: I think this is not necessary. Either we fix from the beginning that they are the same or we allow for them to be different. One may want to use Lum weights but no cut. Instead of setting Lcut to 0 we should allow to set band to None to achieve this. 
  

    if completeness=='load':
        comp_band_loaded=completeness_path.split('_')[1]
        if comp_band_loaded!=band:
            raise ValueError('Band used for cut does not match loaded file. Got band=%s but completeness file is %s' %(band, completeness_path))
        if galPosterior == True:
            #raise ValueError('Load completeness can only be used with no galaxy posteriors')
            print('\n!!! Load completeness should only be used with no galaxy posteriors. Be aware that the result can be inconsistent.')
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
                           eventType=eventType,
                    priorlimits=lims ,
                    subset=subset, subset_names=subset_names, 
                    verbose=verbose, level = level, std_number=std_number, zLimSelection=zLimSelection)#compressed=is_compressed)
    
    if do_check_footprint:
        print('Checking DES and GWENS coverage...')
        fp_DES, fp_GWENS = check_footprint(allGW, observingRun)
        print('DES coverage of GW events: fraction of %s %% credible region that falls into DES footprint' %str(100*level))
        print(fp_DES)
        print('GWENS coverage of GW events: fraction of %s %% credible region that falls into GWENS footprint' %str(100*level))
        print(fp_GWENS)
    
    print('Done.')
    
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
    
        cat = GLADE(catalogue, compl, useDirac = not galRedshiftErrors, band=band, Lcut=Lcut, verbose=verbose,
              galPosterior=galPosterior, band_weight=band_weight)
        
    elif catalogue == 'GWENS':
        cat = GWENS('GWENS', compl, useDirac= not galRedshiftErrors, verbose=verbose, galPosterior=galPosterior)

    elif catalogue == 'DES':
        cat = DES('DES', compl, useDirac= not galRedshiftErrors, verbose=verbose, galPosterior=galPosterior)

    else:
        raise NotImplementedError('Galaxy catalogues other than GLADE, GWENS or DES are not supported for the moment. ')
    
    gals.add_cat(cat)
    
    if plot_comp:
    
        mymask = None
        if completeness == 'mask':
            mymask = cat._completeness._mask
        
        plot_completeness(out_path, allGW, cat, lims, mask=mymask, verbose=verbose)
    
    print('Done.')
    
    if select_events:
    
        evSelector = EventSelector(gals, completnessThreshCentral)
        
    else:
    
        evSelector = SkipSelection()
    
    ###### 
    # GWgal
    ######
    myGWgal = GWgal(gals, allGW, evSelector,
                    MC=MChom, nHomSamples=nHomSamples, 
                    verbose=verbose, galRedshiftErrors=galRedshiftErrors, zR=zR)
    #myGWgal._select_events(completnessThreshAvg=completnessThreshAvg, completnessThreshCentral=completnessThreshCentral)
    
    
    ###### 
    # Grids
    ######
    if goalParam=='H0':
        assert Xi0min==Xi0max
        H0grid=np.linspace(lims.H0min, lims.H0max, nPointsPosterior)
        Xi0grid=Xi0min
        grid=H0grid
        print('Fiducial value of Xi0 fixed to %s ' %Xi0grid)
    elif goalParam=='Xi0':
        assert H0min==H0max
        H0grid= H0min #H0GLOB
        Xi0grid=np.linspace(lims.Xi0min, lims.Xi0max, nPointsPosterior)
        grid=Xi0grid
        print('Fiducial value of H0 fixed to %s ' %H0grid)
    np.savetxt(os.path.join(out_path, goalParam+'_grid.txt'), grid)
    
    
    ###### 
    # Beta
    ######
    if do_inference:
        print('\n-----  COMPUTING BETAS ....')
        betas = beta_case(which_beta, myGWgal.selectedGWevents, lims, H0grid, Xi0grid, evSelector, gals, massDist, lamb) #which_beta, allGW, lims, H0grid, Xi0grid, EventSelector, gals
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
    
        post, post_cat, post_compl = plot_post(out_path, grid, post, post_cat, post_compl, list(myGWgal.selectedGWevents.keys()),
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
    #else:
    #    myGWgal.select_gals()
        
    myGWgal._get_summary()
    #summary = myGWgal.summary()
    myGWgal.summary.to_csv(os.path.join(out_path, 'summary.csv') )
    
    print('\nDone in %.2fs' %(time.time() - in_time))
    
    sys.stdout = sys.__stdout__
    myLog.close()    
    
    





if __name__=='__main__':
    main()
