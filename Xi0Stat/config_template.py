#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:51:08 2020

@author: Michi
"""
from globals import H0GLOB, Xi0Glob

####################################
# Set here the parameters of the run 
####################################


# --------------------------------------------------------------
# INFERENCE SETUP
# --------------------------------------------------------------
do_inference=True

## Variable H0 or Xi0
goalParam = 'H0'

## Output folder name
fout = 'O2_run1'

## Prior limits
Xi0min =  Xi0Glob # 0.3 #Xi0Glob
Xi0max =  Xi0Glob # 10
H0min =   20 # H0GLOB
H0max =    140


## Number of points for posterior grid
nPointsPosterior = 50

verbose=True

# --------------------------------------------------------------
# GW DATA OPTIONS
# --------------------------------------------------------------


## Select dataset : O2, O3
observingRun = 'O2'

## Select BBH or BNS
eventType='BBH'

# How to select credible region in redshift, 'skymap' or 'header'
zLimSelection='skymap'

## Names of events to analyse. If None, all events in the folder will be used
subset_names =  None #['GW190425',]


## Select events based on completeness at the nominal position
select_events=True

## Threshold in probability at position of the event, for event selection
completnessThreshCentral=0.1

# THIS IS NOT USED FOR THE MOMENT!
# completnessThreshAvg=0.01

## Confidence region for GW skymaps
level = 0.97
std_number=None # if none, it is computed from level


# --------------------------------------------------------------
# GALAXY CATALOGUE OPTIONS
# --------------------------------------------------------------

## Galaxy catalogue
catalogue='GLADE'

# Check if events fall in DES or GWENS footprint
do_check_footprint=False


## Luminosity cut and band. 
# Band should be None if we use number counts
Lcut=0.6
# Band for lum cut
band='B' #'B' # B, K, or None . 
# Average galaxy density in comoving volume, used if band='None'
Nbar = 0.1
# Band for lum weights
band_weight = band #'B' # B, K, or None . 



## Use of galaxy redshift errors
galRedshiftErrors = True

## Use of galaxy posteriors
galPosterior = True


# --------------------------------------------------------------
# COMPLETENESS AND COMPLETION OPTIONS
# --------------------------------------------------------------

## Completeness. 'load', 'pixel', 'mask', 'skip'
completeness = 'mask'
# path of completeness file if completeness='load'
completeness_path = 'hpx_B_zmin0p01_zmax0p25_nside32_npoints25.txt'
# Options for SuperPixelCompleteness
angularRes, interpolateOmega = 4, False
zRes = 30
# nMasks for mask completeness. 2 for DES/GWENS, >5 for GLADE
nMasks = 9
#
plot_comp=False


## Type of completion: 'mult' , 'add' or 'mix' 
completionType = 'mix'
# Use MC integration or not in the computation of additive completion
MChom=True
# N. of homogeneous MC samples
nHomSamples=10000



# --------------------------------------------------------------
# BETA OPTIONS
# --------------------------------------------------------------

## Which beta to use. 'fit', 'MC', 'hom', 'cat'
which_beta = 'MC'
# only used when which_beta='hom'. If 'scale', use individually SNR rescaled dmax estimate. If 'flat' use d of event. If a number use that for all events. 
betaHomdMax = 425.7 
# Max redshift  of the region R,  if beta is 'fit'
zR = 10
# n of MC samples for beta MC
nSamplesBetaMC= 100000
nUseCatalogBetaMC = False
SNRthresh=8
