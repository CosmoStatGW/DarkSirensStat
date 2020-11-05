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


##
do_inference=True

## Variable H0 or Xi0
goalParam = 'H0'

## Output file name
fout = 'O3_GW190425_test'


## Number of points for posterior grid
nPointsPosterior = 50


## Select dataset : O2, O3
observingRun = 'O3'
# True if GW data are saved in .gz
#is_compressed=True

# How to select credible region in redshift
zLimSelection='skymap' #header

## Names of events to analyse. If None, all events in the folder will be used
subset_names = ['GW190425',]# 'GW190620_030421']
# GW170817
# 'GW190814'
#['GW170608', 'GW150914', 'GW170814', 'GW151226']
#['GW151226', 'GW170818', 'GW170608', 'GW170814', 'GW170809', 'GW151012', 'GW170729', 'GW170104', 'GW170823', 'GW150914'] 

## Threshold in average probability and prob. at position of the event
completnessThreshAvg=0.01
completnessThreshCentral=0.1


## Prior limits
Xi0min =  Xi0Glob #0.1 #Xi0Glob
Xi0max =  Xi0Glob #10
H0min =   40 #H0GLOB
H0max =    140


## Galaxy catalogue
catalogue='GLADE'


## Luminosity cut and band. 
# Band should be None if we use number counts
Lcut=0.6
#Band for lum cut
band='B' #'B' # B, K, or None . 
Nbar = 0.1
#Band for lum weights
band_weight ='B' #'B' # B, K, or None . 


## Confidence region for GW skymaps
level = 0.99
std_number=3 # if none, it is computed from level

## Completeness. 'load', 'pixel', 'mask', 'skip'
completeness = 'load'
# path of completeness file if completeness='load'
completeness_path = 'hpx_B_zmin0p01_zmax0p25_nside32_npoints25.txt'
# Options for SuperPixelCompleteness
angularRes, zRes, interpolateOmega = 4, 50, False
# nMasks for mask completeness
nMasks = 2
#
plot_comp=True


## Use of galaxy redshift errors
galRedshiftErrors = True

## Use of galaxy posteriors
galPosterior = True
# Use dirac deltas to compute compleness and gaalxy posteriors
useDirac=False


## Type of completion: 'mult' , 'add' . Default is mixed
completionType = 'add'
# Use MC integration or not in the computation of additive completion
MChom=False
# N. of homogeneous MC samples
nHomSamples=10000



## Which beta to use. 'fit', 'MC', 'hom', 'cat'
which_beta = 'fit'
# Max redshift  of the region R,  if beta is 'fit'
zR = 0.5
# n of MC samples for beta MC
nMCSamplesBeta = 1000000
SNRthresh=8
