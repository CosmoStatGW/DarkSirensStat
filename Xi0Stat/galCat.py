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
from Xi0Stat.keelin import bounded_keelin_3_discrete_probabilities


class GalCat(ABC):
    
    def __init__(self, foldername, completeness, useDirac, verbose, **kwargs):
        
        self._path = os.path.join(dirName, 'data', foldername)
          
        self._nside = 128
        self._useDirac = useDirac
        self.data = pd.DataFrame()
        self.verbose = verbose
        
        self.load(**kwargs)
        
        self.selectedData = self.data
        
        self._completeness = deepcopy(completeness)
        self._completeness.verbose = verbose
        self._completeness.compute(self.data, useDirac)
        
    def get_data(self):
        return self.selectedData
        
    def select_area(self, pixels, nside):
        print("Restricting area of the catalogue to %s pixels with nside=%s" %(pixels.shape[0], nside))
        pixname = "pix" + str(nside)
        
        if not pixname in self.data:
            self.data.loc[:, pixname] = hp.ang2pix(nside, self.data.theta, self.data.phi)

        mask = self.data.isin({pixname: pixels}).any(1)

        self.selectedData = self.data[mask]
        print('%s galaxies kept' %self.selectedData.shape[0])
        
    def set_z_range_for_selection(self, zMin, zMax):
        print("Setting z range of the catalogue between %s, %s" %(zMin, zMax))
        self.selectedData = self.selectedData[(self.selectedData.z >= zMin) & (self.selectedData.z < zMax)]
        print('%s galaxies kept' %self.selectedData.shape[0])
        
        
    @abstractmethod
    def load(self):
        pass
    
    def completeness(self, theta, phi, z, oneZPerAngle=False):
        return self._completeness.get(theta, phi, z, oneZPerAngle)


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
        
        

        #print('Correcting %s for group velocities...' %which_z)

#        df_groups.loc[:, 'isInCat'] = df_groups['PGC'].isin(df['PGC'])
#        print(df_groups)
#        df_groups.set_index(keys=['PGC1'], drop=False, inplace=True)
#        groups = df_groups.groupby(level=0)
#        #groups = df_groups.groupby('PGC1')
#
#        isModified = np.zeros(len(df), dtype=int)
#
#        for grname, gr in groups:
#
#            if gr.isInCat.any():
#                print('y')
#                galnames = gr['PGC']
#                mask = df['PGC'].isin(galnames)
#
#                z_group = gr.HRV.mean()/clight
#
#                df.loc[mask, which_z] = z_group
#                isModified = isModified | mask.to_numpy().astype(int)
#
#        df.loc[:, 'group_correction'] = isModified
       
        df.loc[:, which_z+'_or'] = df[which_z].values
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

        
    
    def CMB_correction(self, df, which_z='z_cosmo'):
        
        '''
        Gives cosmological redshift in CMB frame starting from heliocentric
        
        Inputs: df - dataframe to correct
                which_z - name of column to correct
        
        Output : df,  with a new column given by 
                which_z +'_CMB'
        
        '''
        
        #print('Correcting %s for CMB reference frame...' %which_z)
        
        v_gal = clight*df[which_z].values
        phi_CMB, dec_CMB = gal_to_eq(np.radians(l_CMB), np.radians(b_CMB))
        theta_CMB =  0.5 * np.pi - dec_CMB
                        
        delV = v_CMB*(np.sin(df.theta)*np.sin(theta_CMB)*np.cos(df.phi-phi_CMB) +np.cos(df.theta)*np.cos(theta_CMB))
            
        v_corr = v_gal+delV 
            
        z_corr = v_corr/clight
        df.loc[:,which_z+'_CMB'] = z_corr
  
 
    
    
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
    
    def __init__(self, completionType = None, **kwargs):
    
        self._galcats = []
        self._catweights = []
        
        self._additive = False
        self._multiplicative = False
        
        if completionType == 'add':
            self._additive = True
        elif completionType == 'mult':
            self._multiplicative = True
    
    
    def add_cat(self, cat, weight = 1):
        self._galcats.append(cat)
        self._catweights.append(weight)
        
        
    def total_completeness(self, theta, phi, z, oneZPerAngle=False):
    
        # sums completnesses of all catalogs, taking into account the additional
        # catalog weights
        
        res = 0
        for c, w in zip(self._galcats, self._catweights):
            res += w*c.completeness(theta, phi, z, oneZPerAngle)
        
        return res + 1e-9
        #return sum(list(map(lambda c: c.completeness, self._galcats)))
    
    def select_area(self, pixels, nside):
        for c in self._galcats:
            c.select_area(pixels, nside)
            
    def set_z_range_for_selection(self, zMin, zMax):
        for c in self._galcats:
            c.set_z_range_for_selection(zMin, zMax)
    

    def get_inhom_contained(self, zGrid, nside):
        ''' return pixels : array N_galaxies
        
                    weights: array N_galaxies x len(zGrid)
        '''
        
        allpixels = []
        allweights = []
        
        # iterate through catalogs and add results to lists
        
        for c, w in zip(self._galcats, self._catweights):
        
            # shorthand
            d = c.get_data()
            
            pixname = "pix" + str(nside)
            
            # compute this only once
            if not pixname in c.get_data():
                d.loc[:, pixname] = hp.ang2pix(nside, d.theta, d.phi)

            # pixels are already known
            allpixels.append(d[pixname].to_numpy())
            
            # keelin weights. N has to be tuned for speed vs quality
            weights = bounded_keelin_3_discrete_probabilities(zGrid, 0.16, d.z_lower, d.z, d.z_upper, d.z_lowerbound, d.z_upperbound, N=40, P=0.99999)
            
            if weights.ndim == 1:
                weights = weights[np.newaxis, :]
            
            weights *= d.w[:, np.newaxis]
            
            # completness eval for each gal, on grid
            completnesses = c.completeness(d.theta, d.phi, zGrid)
                
            # for additive completion we do not divide by completness as opposed to otherwise (mult, mix)
            # an overall factor of completeness from the weighted average
            # over catalogs then survives (in the case of 1 catalog, it is cancelling with total_completeness!).
            # We get this one more factor of completeness in the case of additive completion automatically from the confidence function, which returns its argument in the case of additive completion.
            # in all other cases (mult, mix), the confidence is a probability of trust in pure multiplicative completion and is =1 in mult, and between 0 and 1 in mix.
            
            weights *= self.confidence(completnesses)
            weights *= w
          
            weights /= self.total_completeness(d.theta, d.phi, zGrid)
           
            weights /= c._completeness._comovingDensityGoal
            
            allweights.append(weights)
            
        #return np.squeeze(np.vstack(allpixels)), np.vstack(allweights)
        return np.hstack(allpixels), np.vstack(allweights)
    

    def get_inhom(self, nside):
        '''
        returns pixels, redshifts and weights of all galaxies (redshift medians) in the selection, ignoring galaxy redshift errror pdfs
        
        returns:
        pixels :   array nGal
        redshifts: array nGal
        weights:   array nGal
    
        '''
        
        allpixels = []
        allredshifts = []
        allweights = []
        
        for c, w in zip(self._galcats, self._catweights):
        
            # shorthand
            d = c.get_data()
            
            pixname = "pix" + str(nside)
            
            # compute this only once
            if not pixname in c.get_data():
                d.loc[:, pixname] = hp.ang2pix(nside, d.theta, d.phi)

            allpixels.append(d[pixname].to_numpy())
            
            
            weights = d.w.to_numpy().copy()
            
            redshifts = d.z.to_numpy()
            allredshifts.append(redshifts)
            
            # completness eval for each gal
            completnesses = c.completeness(d.theta, d.phi, redshifts, oneZPerAngle = True)
                
            weights *= self.confidence(completnesses)
            
            weights *= w
            
          
            weights /= self.total_completeness(d.theta, d.phi, redshifts, oneZPerAngle = True)
       
            weights /= c._completeness._comovingDensityGoal
            
            allweights.append(weights)
            
        #return np.squeeze(np.vstack(allpixels)), np.squeeze(np.vstack(allredshifts)), np.squeeze(np.vstack(allweights))
        return np.hstack(allpixels), np.hstack(allredshifts), np.hstack(allweights)
    
    def eval_inhom(self, Omega, z):
        '''
        For the future if we had posterior samples
        '''
        pass
    
    def eval_hom(self, theta, phi, z):
        '''
        Homogeneous completion part. Second term in 2.59
        '''
        assert(len(theta) == len(z))
        
        ret = np.zeros(len(theta))
        
        for c, w in zip(self._galcats, self._catweights):
            
        
            # completness eval for each point
            completnesses = c.completeness(theta, phi, z, oneZPerAngle = True)
            
            # for catalog averaging (1)
            retc = completnesses
            
            # how much of homogeneous stuff to add - note in case of additive completion, confidence returns its argument, and we have 1 - completness, the correct prefactor in that case
            
            retc *= (1-self.confidence(completnesses))
            
            # for catalog averaging (2)
            retc *= w
            
            # homogeneous density
            #   divide by it - do not multiply here.
            #   retc *= c._completeness._comovingDensityGoal
            
            ret += retc
        
        # for catalog averaging (3)
        ret /= self.total_completeness(theta, phi, z, oneZPerAngle=True)
        return ret
        
        
    def confidence(self, compl):
    
        if self._multiplicative:
            return 1
        elif self._additive:
            return compl
        else: #interpolation between multiplicative and additive
            confpower = 0.10
            complb = np.clip(compl, a_min=2e-3, a_max=1)
            return np.exp(confpower*(1-1/complb))
