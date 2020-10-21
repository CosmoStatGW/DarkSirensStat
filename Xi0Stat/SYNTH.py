
####
# This object is a quick to generate synthetic galaxy cataloge for testing purposes
####


import pandas as pd
import healpy as hp
import numpy as np

import os, os.path

from Xi0Stat.galCat import GalCat
from Xi0Stat.keelin import convolve_bounded_keelin_3, sample_bounded_keelin_3, fit_bounded_keelin_3

from Xi0Stat.globals import *

class SYNTH(GalCat):
    
    def __init__(self, compl, useDirac, verbose=False, **kwargs):
        
        GalCat.__init__(self, '', compl, useDirac, verbose, **kwargs)
    
    def load(self, zMax = 0.1, zErrFac = 1, comovingNumberDensityGoal = 0.1):
        
        if self.verbose:
            print('Sampling homogenously distributed galaxies...')
        from astropy.cosmology import FlatLambdaCDM
        fiducialcosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
        
        vol = 4*np.pi*fiducialcosmo.comoving_distance(zMax).value**3/3
        nGals = np.int(vol*comovingNumberDensityGoal)
        
        dg = pd.DataFrame(columns=['','theta','phi', 'z', 'z_err', 'z_lowerbound', 'z_lower', 'z_upper', 'z_upperbound'])
        
        dg.loc[:,"theta"] = np.arccos(1-2*np.random.uniform(size=nGals))
        dg.loc[:,"phi"]   = 2*np.pi*np.random.uniform(size=nGals)

        # the following calculation of redshifts is independent of H0
        dmax = fiducialcosmo.comoving_distance(zMax).value
        # sample d_com^2 dd_com from 0 to dmax. CDF is p = d^3/dmax^3, quantile func is dmax*(p**(1/3))
        ds = dmax*np.random.uniform(size=nGals)**(1./3)
         
        z_table = np.linspace(0, zMax, 1000)
        d_table = fiducialcosmo.comoving_distance(z_table).value
        from scipy import interpolate
        redshFromdcom = interpolate.interp1d(d_table, z_table, kind='cubic')

        dg.loc[:,"z"]   = redshFromdcom(ds)
        
        # remove some galaxies
        
        def compl_of_z(z, steep=5/zMax, zstar=0.4*zMax): return 0.5*(1-np.tanh((z-zstar)*steep))
        
        n = 100
        zedges = np.linspace(0,zMax,n)
        
        if self.verbose:
            print('Removing galaxies according to prescribed incompleteness function...')
        for i, z in enumerate(zedges):
            if i == n-1:
                z2 = zMax+0.0001
            else:
                z2 = zedges[i+1]
            
            mask = (dg.theta.to_numpy() < np.pi/2) & (dg.z.to_numpy() <= z2) & (dg.z.to_numpy() > z)

            nRemove = np.int(np.sum(mask)*(1-compl_of_z(z+0.5*zMax/n)))
            
            # indices of all relevant galaxies in this volume
            idx = np.nonzero(mask)[0]
            
            if nRemove > 0:
                
                # sample nRemove *unique* elements from idx
                #import random
                #idxidx = random.sample(range(len(idx)), nRemove)
                #idxDrop = idx[idxidx]
                np.random.shuffle(idx)
                idxDrop = set(idx[:nRemove])
                
                # remove them - anecdocially, taking is faster than dropping
                
                idxKeep = set(range(dg.shape[0])) - idxDrop
                dg = dg.take(list(idxKeep))
            
        
        dg.loc[:,"w"] = 1
        
        if self.verbose:
            print("Sample observations from truncated gaussian likelihood...")
        zerr = np.ones(len(dg.z))
        #zerr = dg.loc[:,"z"].to_numpy()/4
        dg.loc[:,"z_err"]   = zErrFac*np.clip(zerr, a_min = None, a_max = 0.01)
        dg.loc[:,"z"] = sample_trunc_gaussian(mu = dg.loc[:,"z"], sigma = dg.loc[:,"z_err"], lower = 0, size=1)
        
        ####
        from astropy.cosmology import FlatLambdaCDM
        fiducialcosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
        area = 2*np.pi
        zmax1 = 1.0001*np.max(dg[(dg.theta.to_numpy() < np.pi/2) ].z.to_numpy())
        zmax2 = 1.0001*np.max(dg[(dg.theta.to_numpy() > np.pi/2) ].z.to_numpy())
        self.zedges1 = np.linspace(0, zmax1, 90+1)
        self.zedges2 = np.linspace(0, zmax2, 90+1)
        z1 = self.zedges1[:-1]
        z2 = self.zedges1[1:]
        vol1 = area * (fiducialcosmo.comoving_distance(z2).value**3 - fiducialcosmo.comoving_distance(z1).value**3)/3
        self.zcenters1 = 0.5*(z1 + z2)
        z1 = self.zedges2[:-1]
        z2 = self.zedges2[1:]
        self.zcenters2 = 0.5*(z1 + z2)
        vol2 = area * (fiducialcosmo.comoving_distance(z2).value**3 - fiducialcosmo.comoving_distance(z1).value**3)/3
        coarseden1 = np.zeros(vol1.shape)
        coarseden2 = np.zeros(vol2.shape)
        
        ###
        
        if not self._useDirac:
        
            dg.loc[:,"z_lower"] = dg.loc[:,"z"]
            dg.loc[:,"z_lowerbound"] = dg.loc[:,"z"]
            dg.loc[:,"z_upper"] = dg.loc[:,"z"]
            dg.loc[:,"z_upperbound"] = dg.loc[:,"z"]
            
            L = 0
            block = 10000
            
            if self.verbose:
                print("Computing galaxy posteriors...")
                
            while True:
                
                R = L + block
             
                
                
                # evaluate likelihood at fixed observations on sensible grids in mu
                # note that sigma depends on mu as well.
                
                if R >= len(dg):
                    lowerbound = dg.z.to_numpy()[L:] - 7*dg.z_err.to_numpy()[L:]
                    lowerbound[lowerbound<0] = 0
                    upperbound = dg.z.to_numpy()[L:] + 7*dg.z_err.to_numpy()[L:]
                else:
                    lowerbound = dg.z.to_numpy()[L:R] - 7*dg.z_err.to_numpy()[L:R]
                    lowerbound[lowerbound<0] = 0
                    upperbound = dg.z.to_numpy()[L:R] + 7*dg.z_err.to_numpy()[L:R]
                
                #lowerbound = np.zeros(lowerbound.shape)
                #upperbound = np.ones(upperbound.shape)*0.1
                mugrid = np.linspace(lowerbound, upperbound, 400).T
                # remove leading zeros (if lowerbound = 0), sigma=0 would follow and is ill-defined
                mugrid = mugrid[:,1:]
                #  copy the algorithm to compute the error from what used to be mu
                # (despite being called z), the true redshift
                #sigmagrid = mugrid/4
                sigmagrid = np.ones(mugrid.shape)
                sigmagrid = zErrFac*np.clip(sigmagrid, a_min = None, a_max = 0.01)
                
                # fix observation to dg.z, eval as function of truth mu
                
                if R >= len(dg):
                    pdfs = trunc_gaussian_pdf(dg.loc[:,"z"].to_numpy()[L:], mu = mugrid, sigma=sigmagrid, lower=0)
                else:
                    pdfs = trunc_gaussian_pdf(dg.loc[:,"z"].to_numpy()[L:R], mu = mugrid, sigma=sigmagrid, lower=0)
                    
                # multiply by prior
                #pdfs *= mugrid**2
                
    
                rsqdrdz = fiducialcosmo.comoving_distance(mugrid).value**2 / fiducialcosmo.H(mugrid).value
                pdfs *= rsqdrdz
              #  pdfs *= mugrid**2
                # fit keelin pdfs to this posterior. normalization is not necessary
               
                
                
                ####
                if R >= len(dg):
                    mask1 = (dg.theta.to_numpy()[L:] < np.pi/2)
                    mask2 = (dg.theta.to_numpy()[L:] > np.pi/2)
                else:
                    mask1 = (dg.theta.to_numpy()[L:R] < np.pi/2)
                    mask2 = (dg.theta.to_numpy()[L:R] > np.pi/2)
                
                # actually prior is not constant in comoving for the part that drops!
                pdfs[mask1, :] = pdfs[mask1, :]*compl_of_z(mugrid[mask1, :])
                
            
                cdfs = np.cumsum(pdfs, axis=1)
                pdfs = pdfs / ((cdfs[:, -1])[:, np.newaxis])
                cdfs =  cdfs / ((cdfs[:, -1])[:, np.newaxis])
                
                fits = fit_bounded_keelin_3(0.16, grids=mugrid, pdfs=pdfs)
                   
                   
                for i in range(len(self.zcenters2)):
                    maskz = (self.zedges1[i] < mugrid) & (mugrid < self.zedges1[i+1])
                    mask = mask1[:,np.newaxis] & maskz
                    coarseden1[i] += np.sum( pdfs[mask] )/ vol1[i]
                    maskz = (self.zedges2[i] < mugrid) & (mugrid < self.zedges2[i+1])
                    mask = mask2[:,np.newaxis] & maskz
                    coarseden2[i] += np.sum( pdfs[mask] )/ vol2[i]
                ###
                
                
                
                if R >= len(dg):
            
                    dg.iloc[L:, dg.columns.get_loc("z_lowerbound")] = fits[:, 0]
                    dg.iloc[L:, dg.columns.get_loc("z_lower")] = fits[:, 1]
                    dg.iloc[L:, dg.columns.get_loc("z")] = fits[:, 2]
                    dg.iloc[L:, dg.columns.get_loc("z_upper")] = fits[:, 3]
                    dg.iloc[L:, dg.columns.get_loc("z_upperbound")] = fits[:, 4]
                    break
                else:
                    dg.iloc[L:R, dg.columns.get_loc("z_lowerbound")] = fits[:, 0]
                    dg.iloc[L:R, dg.columns.get_loc("z_lower")] = fits[:, 1]
                    dg.iloc[L:R, dg.columns.get_loc("z")] = fits[:, 2]
                    dg.iloc[L:R, dg.columns.get_loc("z_upper")] = fits[:, 3]
                    dg.iloc[L:R, dg.columns.get_loc("z_upperbound")] = fits[:, 4]
#                dg.z[L:R] = fits[:, 2]
#                dg.z_lower[L:R]   = fits[:,1]
#                dg.z_upper[L:R]   = fits[:,3]
#                dg.z_lowerbound[L:R]   = fits[:,0]
#                dg.z_upperbound[L:R]   = fits[:,4]

        #        mask = dg.z_lower < 1e-5
        #        dg.loc[mask, "z_lower"] = dg.z[mask]*0.5
        #        dg.loc[mask, "z_lowerbound"] = 0.0
            
                
                L += block
       
        self.compl1 = coarseden1/0.05
        self.compl2 = coarseden2/0.05
        
        dg.loc[:,"pix"  + str(self._nside)]   = hp.ang2pix(self._nside, dg.theta, dg.phi)
                
        self.data = self.data.append(dg, ignore_index=True)
            
            
         
            
            

