#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:36:19 2020

@author: Michi
"""
from globals import *
from abc import ABC, abstractmethod

class Selector(ABC):
    
    def __init__(self, completnessThreshCentral, completnessThreshAvg=0.,
                 select_events=True, select_gals=False,):
        self.completnessThreshCentral=completnessThreshCentral
        self.completnessThreshAvg=completnessThreshAvg
        self.select_events=select_events
        self.select_gals=select_gals
    
    @abstractmethod
    def is_good(self):
        pass

    @abstractmethod
    def is_good_event(self):
        pass



class SkipSelection(Selector):
    
    def __init__(self):
        completnessThreshCentral=0.
        completnessThreshAvg=0.
        select_events=False
        select_gals=False
        Selector.__init__(self, completnessThreshCentral, completnessThreshAvg=completnessThreshAvg, select_events=select_events, select_gals=select_gals)
        
    def is_good(self,  theta, phi, dist_true, completeness):
        return np.repeat(True, theta.shape)
    
    def is_good_event(self, GWevent, completeness):
        return True


class EventSelector(object):
    
    
    def __init__(self, completnessThreshCentral, 
                 completnessThreshAvg=0.,
                 select_events=True, select_gals=False,
                 H0=None, Xi0=None, n=None):
        
        #self.completnessThreshCentral=completnessThreshCentral
        #self.completnessThreshAvg=completnessThreshAvg
        Selector.__init__(self, completnessThreshCentral, completnessThreshAvg=completnessThreshAvg, select_events=select_events, select_gals=select_gals)
        #self.select_events=select_events
        #self.select_gals=select_gals
        
        
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
        
    
    def is_good(self, theta, phi, dist_true, completeness):
        '''
        Returns mask (  nSamples x  1) 
        
        Checks if dist is in region of excluded events
        
        '''
        
        zz = z_from_dLGW_fast(dist_true, H0=self.H0, Xi0=self.Xi0, n=self.n)
        
        Pc = completeness(theta, phi, zz, oneZPerAngle=True) #gals.total_completeness(theta, phi, zz)
        
        return Pc>self.completnessThreshCentral
    
    
    def is_good_event(self, GWevent, completeness):
        
        PcEv = completeness( *GWevent.find_event_coords(polarCoords=True), GWevent.zEv, oneZPerAngle=True) #gals.total_completeness( *GWevent.find_event_coords(polarCoords=True), GWevent.zEv)
        
        return PcEv > self.completnessThreshCentral
        
        