#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:36:19 2020

@author: Michi
"""
from globals import *
from abc import ABC, abstractmethod

class Selector(ABC):
    

    def __init__(self):
        pass

    @abstractmethod
    def is_good(self):
        pass

    @abstractmethod
    def is_good_event(self):
        pass



class SkipSelection(Selector):
    
    def __init__(self):
        pass
        
    def is_good(self,  theta, phi, dist_true):
        return np.repeat(True, theta.shape)
    
    def is_good_event(self, GWevent):
        return True


class EventSelector(Selector):
    
    
    def __init__(self, gals, completnessThreshCentral,
                 completnessThreshAvg=0.,
                 select_events=True, select_gals=False,
                 H0=None, Xi0=None, n=None):
                 
        self.completnessThreshCentral=completnessThreshCentral
        self.completnessThreshAvg=completnessThreshAvg
        self.select_events=select_events
        self.select_gals=select_gals
       
        self.completenessFunc = gals.total_completeness
        
        if H0 is None:
            self.H0=H0GLOB
        else: self.H0=H0
        if Xi0 is None:
            self.Xi0=Xi0Glob
        else: self.Xi0=Xi0
        if n is None:
            self.n=nGlob
        else:
            self.n=n
        
    
    def is_good(self, theta, phi, dist_true):
        '''
        Returns mask (  nSamples x  1) 
        
        Checks if dist is in region of excluded events
        
        '''
        
        zz = z_from_dLGW_fast(dist_true, H0=self.H0, Xi0=self.Xi0, n=self.n)

        if np.isscalar(theta):
            Pc = self.completenessFunc(theta, phi, zz.item())
        else:
            Pc = self.completenessFunc(theta, phi, zz, oneZPerAngle=True)
        
        return Pc>self.completnessThreshCentral
    
    
    def is_good_event(self, GWevent):
        
        dL, _, _, _ = GWevent.find_r_loc(verbose=False)
        
        return self.is_good(*GWevent.find_event_coords(polarCoords=True), dL)
        
        
