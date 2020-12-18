#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 09:59:52 2020

@author: Michi
"""

from globals import *


# --------------------------------------------------------------
# INFERENCE SETUP
# --------------------------------------------------------------
do_inference=True

## Variable H0 or Xi0
goalParam = 'H0'

## Output folder name
fout = 'O3_H0_MCbeta'

## Prior limits
Xi0min = Xi0Glob #Xi0Glob # 0.3 #Xi0Glob
Xi0max =  Xi0Glob #Xi0Glob # 10
H0min =   20#20 # H0GLOB
H0max =   140 #140


## Number of points for posterior grid
nPointsPosterior = 100

verbose=True


## Select dataset : O2, O3
observingRun = 'O3'

# --------------------------------------------------------------


## O2 : IMRPhenomPv2NRT_lowSpin  IMRPhenomPv2NRT_highSpin
## O3 : C01:IMRPhenomPv3HM C01:NRSur7dq4

wf_model= 'C01:IMRPhenomPv3HM'  #IMRPhenomPv2NRT_highSpin

# Area of the cone around the position of the event, in deg^2
cone_area = 0.0045



## Only MC o hom
which_beta='MC'

SNRthresh=8
nSamplesBetaMC= 200000

betaHomdMax= 400 #425.7 600
zR=2

