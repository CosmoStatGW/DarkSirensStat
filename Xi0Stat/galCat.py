#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:32:31 2020

@author: Michi
"""

####
# This module contains a abstract classes to handle a galaxy catalogue
####

import pandas as pd
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from copy import deepcopy

from Xi0Stat.globals import *


class GalCat(ABC):
    
    def __init__(self, foldername, completeness, **kwargs):
        print('Initializing GalCat...')
        
        self._path = os.path.join(dirname, 'data', foldername)
        
        self._nside = 128
        
        self.data = pd.DataFrame()
        
        self.load(**kwargs)
        
        self._completeness = deepcopy(completeness)
        self._completeness.compute() #self.data
        
    
    @abstractmethod
    def load(self):
        pass
    
    def completeness(self, Omega, z):
        return self._completeness.get(Omega, z)
    
        
    def completeness_maps(self):
    
        zs = np.linspace(0.00, 0.5, 40)
        nPix = hp.nside2npix(512)
        px = np.arange(nPix)
        
        finemaps = self.completeness(hp.pix2ang(512, px), zs)
        
        for z in zs:
            
            finemap = self.completeness(hp.pix2ang(512, px), z)
                                        
            hp.mollview(finemap, notext=True)
            
            
            plt.savefig('z=' + str(z) + '.png')


    def group_correction(self, df, df_groups, which_z='z_cosmo'):
        '''
        Corrects cosmological redshift in heliocentric frame 
        for peculiar velocities in galaxies 
        inside the group galaxy catalogue in  arXiv:1705.08068, table 2
        To be applied BEFORE changing to CMB frame
        
        Inputs: df - dataframe to correct
                df groups - dataframe of group velocities
                which_z - name of column to correct
        
        Output : df, but with the column given by which_z corrected for peculiar velocities
                    in the relevant cases and a new column named which_z+'_or' 
                    with the original redshift
        
        '''
        
        df.loc[:, which_z+'_or'] = df[which_z].values        
            
        print('Correcting %s for group velocities...' %which_z)
   
            
        zs = df.loc[df['PGC'].isin(df_groups['PGC'])][['PGC', which_z]] 
        z_corr_arr = []
        #z_group_arr = []
        for PGC in zs.PGC.values:
                #z_or=zs.loc[zs['PGC']==PGC][which_z_correct].values[0]
            PGC1=df_groups[df_groups['PGC']==PGC]['PGC1'].values[0]
                #print(PGC1)
                
            z_group = df_groups[df_groups['PGC1']== PGC1].HRV.mean()/clight
                
                
            z_corr_arr.append(z_group)
        z_corr_arr=np.array(z_corr_arr)
            
        df.loc[df['PGC'].isin(df_groups['PGC']), which_z] = z_corr_arr                        
        correction_flag_array = np.where(df[which_z+'_or'] != df[which_z], 1, 0)
        df.loc[:, 'group_correction'] = correction_flag_array
            
        return df
        
    
    def CMB_correction(self, df, which_z='z_cosmo'):
        
        '''
        Gives cosmological redshift in CMB frame starting from heliocentric
        
        Inputs: df - dataframe to correct
                which_z - name of column to correct
        
        Output : df,  with a new column given by 
                which_z +'_CMB'
        
        '''
        
        print('Correcting %s for CMB reference frame...' %which_z)
        
        v_gal = clight*df[which_z].values
        phi_CMB, dec_CMB = gal_to_eq(np.radians(l_CMB), np.radians(b_CMB))
        theta_CMB =  0.5 * np.pi - dec_CMB
                        
        delV = v_CMB*(np.sin(df.theta)*np.sin(theta_CMB)*np.cos(df.phi-phi_CMB) +np.cos(df.theta)*np.cos(theta_CMB))
            
        v_corr = v_gal+delV 
            
        z_corr = v_corr/clight
        df.loc[:,which_z+'_CMB'] = z_corr
        return df
 
    
    
def gal_to_eq(l, b):
    '''
    input: galactic coordinates (l, b) in radians
    returns equatorial coordinates (RA, dec) in radians
    
    https://en.wikipedia.org/wiki/Celestial_coordinate_system#Equatorial_↔_galactic
    '''
    
    l_NCP = np.radians(122.93192)
    
    del_NGP = np.radians(27.128336)
    alpha_NGP = np.radians(192.859508)
    
    
    RA = np.arctan((np.cos(b)*np.sin(l_NCP-l))/(np.cos(del_NGP)*np.sin(b)-np.sin(del_NGP)*np.cos(b)*np.cos(l_NCP-l)))+alpha_NGP
    dec = np.arcsin(np.sin(del_NGP)*np.sin(b)+np.cos(del_NGP)*np.cos(b)*np.cos(l_NCP-l))
    
    return RA, dec



    
class GalCompleted(object):
    
    def __init__(self, **kwargs):
        print('Initializing GalCompleted...')
        self._galcats = []
    
    def add_cat(self, cat):
        self._galcats.append(cat)
        
    def total_completeness(self, Omega, z):
        return sum(list(map(lambda c: c.completeness(Omega, z), self._galcats)))
    
    def confidence(compl):
        # multiplicative completion:
        #return 1
        # homogeneous completion
        #return 0
        # general interplation
        confpower = 0.05
        return np.exp(confpower*(1-1/compl))
        
    def get_inhom_contained(self, Omega, z):
        pass
    
    def eval_inhom(self, Omega, z):
        pass
        
    def eval_hom(self, Omega, z):
        pass
