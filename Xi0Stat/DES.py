
####
# This module contains a class to handle the GWENS catalogue
####


import pandas as pd
import healpy as hp
import numpy as np

import os, os.path

from galCat import GalCat


class DES(GalCat):
    
    def __init__(self, foldername, compl, useDirac, verbose = False, **kwargs):
        print('Loading DES...')
            
        GalCat.__init__(self, foldername, compl, useDirac, verbose, **kwargs)

    def load(self, remove_unreliable=True, add_distrib_redshifts=True, galPosterior = True):
        from astropy.io import fits
        filepath = os.path.join(self._path, 'y1a1.fits')
        f = fits.open(filepath)
        
        #dg = pd.DataFrame(data=hp.read_map(filepath,field=[1],verbose=self.verbose), columns=['ra','dec'])
        RA = f[1].data['RA'] 
        DEC = f[1].data['DEC'] 

        dg = pd.DataFrame()

        dg['z'] = f[1].data['DNF_Z']
        dg['z_err'] = f[1].data['DNF_ZSIGMA']
        dg.loc[:,"theta"] = np.pi/2 - (DEC*np.pi/180)
        dg.loc[:,"phi"]   = RA*np.pi/180

        print(len(dg))
        dg = dg[dg.phi.notna()]
        print(len(dg))
        dg = dg[dg.theta.notna()]
        print(len(dg))
        dg = dg[dg.z.notna()]
        print(len(dg))
        dg = dg[dg.z_err.notna()]
        print(len(dg))
        #if CMB_correct:
        #    self.CMB_correction(dg, which_z='z')
        #    dg.loc[:,'z'] = dg.z_CMB

        dg = dg[dg.z > 0]
        print(len(dg))
        dg = dg[dg.z_err > 0]
        print(len(dg))

        if remove_unreliable: #Remove unreliable photo-zs tagged by 99
            dg = dg[dg.z_err < 1.0]
            print(len(dg))
            dg = dg[dg.z_err < 3*dg.z]
            print(len(dg))
            #dg = dg[dg.z_err != 99]

        if add_distrib_redshifts: #add z_lowerbound, z_lower, etc
            dg.loc[dg.z<dg.z_err,'z_err'] = dg.z
            dg.loc[:, 'z_lowerbound'] = dg.z - 3*dg.z_err
            dg.loc[dg.z_lowerbound < 0, 'z_lowerbound'] = 0
            dg.loc[:, 'z_lower'] = dg.z - dg.z_err
            dg.loc[dg.z_lower < 0.5*dg.z, 'z_lower'] = 0.5*dg.z
            dg.loc[:, 'z_upper'] = dg.z + dg.z_err
            dg.loc[:, 'z_upperbound'] = dg.z + 3*dg.z_err
            
            if galPosterior:
                dg = self.include_vol_prior(dg) 

        dg.loc[:, "w"] = np.ones(dg.shape[0])
        dg.loc[:,"pix" + str(self._nside)]   = hp.ang2pix(self._nside, dg.theta, dg.phi)

        self.data = self.data.append(dg, ignore_index=True)
        f.close()

        #dg = dg.loc[:,['theta','phi', 'z', 'z_err', 'z_lowerbound', 'z_lower', 'z_upper', 'z_upperbound', 'w']]
            
            

