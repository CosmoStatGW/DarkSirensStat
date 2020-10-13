####
# This module contains objects to compute the completeness of a galaxy catalogue
####

from abc import ABC, abstractmethod

from Xi0Stat.keelin import bounded_keelin_3_discrete_probabilities_between
from Xi0Stat.globals import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import healpy as hp

class Completeness(ABC):
    
    def __init__(self, **kwargs):
        self._computed = False
        pass
        
    def compute(self, galdata, useDirac = False):
        print('Computing completeness')
        self.compute_implementation(galdata, useDirac)
        self._computed = True
    
    @abstractmethod
    def compute_implementation(self, galdata, useDirac):
        pass
        
    @abstractmethod
    def zstar(self, theta, phi):
        pass
     
    def get(self, theta, phi, z, oneZPerAngle=False):
        assert(self._computed)
        if np.isscalar(z):
            return np.where(z > self.zstar(theta, phi), self.get_at_z_implementation(theta, phi, z), 1)
        else:
            if oneZPerAngle:
                assert(not np.isscalar(theta))
                assert(len(z) == len(theta))
                close = z < self.zstar(theta, phi)
                ret = np.zeros(len(z))
                ret[~close] = self.get_many_implementation(self, theta[~close], phi[~close], z[~close])
                ret[close] = 1
                return ret
            else:
                if len(z) == len(theta):
                    print('Completeness::get: number of redshift bins and number of angles agree but oneZPerAngle is not True. This may be a coincidence, or indicate that the flag should be True')
                
                ret = self.get_implementation(theta, phi, z)
                close = z[np.newaxis, :] < self.zstar(theta, phi)[:, np.newaxis]
                return np.where(close, 1, ret)
    
    # z is scalar (theta, phi may or may not be)
    # useful to make maps at fixed z and for single points
    @abstractmethod
    def get_at_z_implementation(self, theta, phi, z):
        pass
        
    # z, theta, phi are vectors of same length
    @abstractmethod
    def get_many_implementation(self, theta, phi, z):
        pass
    
    # z is grid, we want matrix: for each theta,phi compute *each* z
    @abstractmethod
    def get_implementation(self, theta, phi, z):
        pass
        
  
    
class SkipCompleteness(Completeness):
    
    def __init__(self, **kwargs):
        print('Initializing SkipCompleteness...')
        Completeness.__init__(self, **kwargs)
        pass
        
    def zstar(self, theta, phi):
        return 0
        
    def compute_implementation(self, galdata, useDirac):
        print("SkipCompleteness: nothing to compute")
        
    def get_at_z_implementation(self, theta, phi, z):
        if np.isscalar(theta):
            return 1
        return np.ones(theta.size)
        
    def get_implementation(self, theta, phi, z):
        return np.ones((theta.size, z.size))
    
    def get_many_implementation(self, theta, phi, z):
        return np.ones(theta.size)
        


    
    
class SuperpixelCompleteness(Completeness):
    
    def __init__(self, comovingDensityGoal, angularRes, zRes, interpolateOmega, **kwargs):
    
        assert(angularRes < 7)
        self._nside = 2**angularRes
        self._npix  = hp.nside2npix(self._nside)
        self._pixarea = hp.nside2pixarea(self._nside)
        self._zRes = zRes
        self.zedges = None
        self.zcenters = None
        
        self._interpolateOmega = interpolateOmega
        self._zstar = None
        
        self._map = np.zeros((self._npix, zRes), dtype=np.float)
        
        from astropy.cosmology import FlatLambdaCDM
        
        self._fiducialcosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
        self._comovingDensityGoal = comovingDensityGoal
        
        Completeness.__init__(self, **kwargs)
        
    def zstar(self, theta, phi):
        if self._interpolateOmega:
            return hp.get_interp_val(self._zstar, theta, phi)
        else:
            return self._zstar[hp.ang2pix(self._nside, theta, phi)]
        
    
    def compute_implementation(self, galdata, useDirac):
    
        coarseden = self._map.copy()
        zmax = 1.0001*np.max(galdata.z.to_numpy())
        self.zedges  = np.linspace(0, zmax, self._zRes+1)
        
        z1 = self.zedges[:-1]
        z2 = self.zedges[1:]
        
        self.zcenters = 0.5*(z1 + z2)
        
        galdata.loc[:,'pix'] = hp.ang2pix(self._nside, galdata.theta, galdata.phi)
        galdata.set_index(keys=['pix'], drop=False, inplace=True)
        
        # need to iterate through pixels...
        print('Preparing... ')
        catparts = []
        for i in np.arange(self._npix):
            if (i % np.int(self._npix/30)) == 0:
                print('#', end='', flush=True)
            if i in galdata.index:
                galpixel = galdata.loc[[i]]
                catparts.append(galpixel)
            else:
                catparts.append(pd.DataFrame())
                
            
        print('Computing in parallel... ', flush=True)
        #from multiprocessing import Pool
        #p = Pool(2)
        #from functools import partial
        def g(galpixel):
    
            if len(galpixel) == 0:
                return np.zeros(len(z1))
            elif useDirac:
                res, _ = np.histogram(a=galpixel.z.to_numpy(), bins=self.zedges, weights=galpixel.w.to_numpy())
                return res.astype(float)
            else:
                weights = bounded_keelin_3_discrete_probabilities_between(self.zedges, 0.16, galpixel.z_lower, galpixel.z, galpixel.z_upper, galpixel.z_lowerbound, galpixel.z_upperbound)
                # if there is 1 galaxy only, weights won't be a matrix - fix
                if weights.ndim == 1:
                    weights = weights[np.newaxis, :]
                weights = weights * self.zcenters**2
                # rows can be fully zero if galaxy sits beyond the z grid - avoid division by 0 by adding small epsilon (small cmp to a typical z^2, so e.g. 10^-4)
                weights = weights / ((1e-9 + np.sum(weights, axis=1)[:, np.newaxis]))
                return np.sum(weights * galpixel.w[:, np.newaxis], axis=0)
          
     
        coarseden = np.vstack( parmap(lambda i : g(catparts[i]), range(len(catparts))) )
        
    
        #print('coarseden', coarseden)
        
        
#        for i in np.arange(self._npix):
#
#            if (i % np.int(self._npix/30)) == 0:
#                print('#', end='', flush=True)
#            # too slow
#            #galdata.drop(i, inplace=True)
#
#            if len(catparts[i]) > 0:
#
#                if useDirac:
#                    coarseden[i, :], _ = np.histogram(a=catparts[i].z.to_numpy(), bins=self.zedges, weights=catparts[i].w.to_numpy())
#
#                else:
#                    weights = bounded_keelin_3_discrete_probabilities_between(self.zedges, 0.16, catparts[i].z_lower, catparts[i].z, catparts[i].z_upper, catparts[i].z_lowerbound, catparts[i].z_upperbound)
#                    coarseden[i, :] = np.sum(weights * catparts[i].w[:, np.newaxis], axis=0)
        
        print(' Almost done!')
        
        vol = self._pixarea * (self._fiducialcosmo.comoving_distance(z2).value**3 - self._fiducialcosmo.comoving_distance(z1).value**3)/3
   
        coarseden /= vol
        self._map = coarseden/self._comovingDensityGoal
        print('Final computations for completeness...')
        zFine = np.linspace(0, zmax, 3000)
        zFine = zFine[::-1]
        evals = self.get_implementation(*hp.pix2ang(self._nside, np.arange(self._npix)), zFine)
       
        # argmax returns "first" occurence of maximum, which is True in a boolean array. we search starting at large z due to the flip
        idx = np.argmax(evals >= 1, axis=1)
        # however, if all enries are False, argmax returns 0, which would be the largest redshift, while we want 0 in that case
        self._zstar = np.where(idx == 0, 0, zFine[idx])
        
        print(self._zstar)
        print(idx)
        
    # z is a number
    def get_at_z_implementation(self, theta, phi, z):
    
        from scipy import interpolate
    
        # if only one point, things are pretty clear
        if np.isscalar(theta):
        
            valsAtZs = np.zeros(self._zRes)
        
            if self._interpolateOmega:
                for i in np.arange(self._zRes):
                    valsAtZs[i] = hp.get_interp_val(self._map[:,i], theta, phi)
            else:
                valsAtZs = self._map[hp.ang2pix(self._nside, theta, phi), :]
        
            f = interpolate.interp1d(self.zcenters, valsAtZs, kind='linear', bounds_error=False, fill_value=(1,0))
            
            return f(z)
        
        # many angles, one z
        else:
            
            # the performance-critical case is when theta, phi are long arrays of points and interpolation in Omega is on. Thus, first interpolate maps in redshift, then interpolate in Omega in the single resulting map. For no interpolation in Omega this is still better when Omega is much longer than the map resolution, since only one long lookup is needed (versus interpolating the values for each Omega in various maps in redshift)
       
            f = interpolate.interp1d(self.zcenters, self._map, kind='linear', bounds_error=False, fill_value=(1,0))
                
            self._buf = f(z)
                
            if self._interpolateOmega:
                return hp.get_interp_val(self._buf, theta, phi)
            else:
                return self._buf[hp.ang2pix(self._nside, theta, phi)]
     
     
     
    def get_many_implementation(self, theta, phi, z):
    
        tensorProductThresh = 4000
    
        # even though we only need to compute len(z) results, for this number small enough < tensorProductThresh, we can do a O(len(z)^2) but loopfree algorithm. At 4000 the O(4000^2) is still faster on my machine but not that much anymore. Larger matrices perhaps take too much memory...
        
        if (len(z) < tensorProductThresh):
            
            res = self.get_implementation(theta, phi, z)
            return np.diag(res)
        
        
        # for many points, a loop is almost necessary - only if many points have same pix32 coordinate we could do something better (if no angular interpolation)
        
        ret = np.zeros(len(z))
        for i, (thetai, phii, zi) in enumerate(zip(theta,phi,z)):
            
            ret[i] = self.get_at_z_implementation(thetai, phii, zi)
            
        return ret
   
    def get_implementation(self, theta, phi, z):
     
         # here theta, phi, z are all arrays.
         # angular interpolation could be done at each z bin, to make one vector of values in z for each galaxy, to be interpolated at each individual z. Since the z are different for each galaxy, scipy.interpolate does not vectorize here (neither does numpy.interp)
         # interpolating in z first would mean making a map for each galaxy, which is even worse.
         # therefore, we do not support interpolation in angle here!

        from scipy import interpolate
             
        f = interpolate.interp1d(self.zcenters, self._map, kind='linear', bounds_error=False, fill_value=(1,0))
             
        # different z will be different columns
        # each column is the interpolated map
        buf = f(z)
             
        if self._interpolateOmega:
            ret = np.zeros((len(theta), len(z)))
            # interpolate map by map
            for i in np.arange(len(z)):
                ret[:,i] = hp.get_interp_val(buf[:,i], theta, phi)
            return ret
        else:
            return buf[hp.ang2pix(self._nside, theta, phi), :]


# class MaskCompleteness(Completeness):
  
