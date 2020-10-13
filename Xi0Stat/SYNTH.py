
####
# This object is a quick to generate synthetic galaxy cataloge for testing purposes
####


import pandas as pd
import healpy as hp
import numpy as np

import os, os.path

from Xi0Stat.galCat import GalCat


class SYNTH(GalCat):
    
    def __init__(self, compl, useDirac, **kwargs):
        print('Creating synthetic catalog...')
        
        GalCat.__init__(self, '', compl, useDirac, **kwargs)
    
    def load(self, zmax = 0.1, comovingNumberDensityGoal=0.1):
       
        print('Sampling homogenously distributed galaxies...')
        from astropy.cosmology import FlatLambdaCDM
        fiducialcosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
        
        vol = 4*np.pi*fiducialcosmo.comoving_distance(zmax).value**3/3
        nGals = np.int(vol*comovingNumberDensityGoal)
        
        dg = pd.DataFrame(columns=['','theta','phi', 'z', 'z_err', 'z_lowerbound', 'z_lower', 'z_upper', 'z_upperbound'])
        
        dg.loc[:,"theta"] = np.arccos(1-2*np.random.uniform(size=nGals))
        dg.loc[:,"phi"]   = 2*np.pi*np.random.uniform(size=nGals)

        # the following calculation of redshifts is independent of H0
        dmax = fiducialcosmo.comoving_distance(zmax).value
        # sample d_com^2 dd_com from 0 to dmax. CDF is p = d^3/dmax^3, quantile func is dmax*(p**(1/3))
        ds = dmax*np.random.uniform(size=nGals)**(1./3)
         
        z_table = np.linspace(0, zmax, 1000)
        d_table = fiducialcosmo.comoving_distance(z_table).value
        from scipy import interpolate
        redshFromdcom = interpolate.interp1d(d_table, z_table, kind='cubic')

        dg.loc[:,"z"]   = redshFromdcom(ds)
        
        # remove some galaxies
        
        def compl_of_z(z, steep=5/zmax, zstar=0.4*zmax): return 0.5*(1-np.tanh((z-zstar)*steep))
        
        n = 100
        zedges = np.linspace(0,zmax,n)
        
        print('Removing galaxies according to prescribed incompleteness function...')
        for i, z in enumerate(zedges):
            if i == n-1:
                z2 = zmax+0.0001
            else:
                z2 = zedges[i+1]
            
            mask = (dg.theta.to_numpy() < 1) & (dg.phi.to_numpy() < np.pi) & (dg.z.to_numpy() <= z2) & (dg.z.to_numpy() > z)

            nRemove = np.int(np.sum(mask)*(1-compl_of_z(z+0.5*zmax/n)))
            
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
        
        zerr = dg.loc[:,"z"].to_numpy()/4
        dg.loc[:,"z_err"]   = np.clip(zerr, a_min = None, a_max = 0.01)
        
        dg.loc[:,"z_lower"]   = dg.z - dg.z_err # *(1+2*dg.z_err/dg.z)
        dg.loc[:,"z_upper"]   = dg.z + dg.z_err
        dg.loc[:,"z_lowerbound"]   = dg.z - 3*dg.z_err # *(1+2*dg.z_err/dg.z)
        dg.loc[:,"z_upperbound"]   = dg.z + 3*dg.z_err

        mask = dg.z_lower < 1e-5
        dg.loc[mask, "z_lower"] = dg.z[mask]*0.5
        dg.loc[mask, "z_lowerbound"] = 0.0
        
        dg.loc[:,"pix"  + str(self._nside)]   = hp.ang2pix(self._nside, dg.theta, dg.phi)
            
        self.data = self.data.append(dg, ignore_index=True)
        
            
         
            
            

