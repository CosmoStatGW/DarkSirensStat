
####
# This module contains a class to handle the GWENS catalogue
####


import pandas as pd
import healpy as hp
import numpy as np

import os, os.path

from galCat import GalCat


class GWENS(GalCat):
    
    def __init__(self, foldername, compl, useDirac, patches=[], verbose = False, **kwargs):
        print('Loading GWENS...')
       
        self._patches = patches
            
        GalCat.__init__(self, foldername, compl, useDirac, verbose, **kwargs)
    
    def load(self):

        if self._patches == []:
            filenames = [os.path.join(self._path, name) for name in os.listdir(self._path) if (os.path.isfile(os.path.join(self._path, name)) and '.csv.gz' in name) ]
        else:
            filenames = [os.path.join(self._path, 'ra_{:03d}_{:03d}.csv.gz'.format(i*15, (i+1)*15)) for i in self._patches]
        print(filenames)   
        for i, filename in enumerate(filenames):
        
            print('Loading patch ' + str(i) + ' of ' + str(len(filenames)) + ' patches of GWENS...')
            
            dg = pd.read_csv(filename, compression='gzip', error_bad_lines=False)
            
            dg = dg.loc[:,['ra','dec', 'medval', 'neg2sig', 'neg1sig', 'pos1sig', 'pos2sig', 'specflag', 'lstar']]

            dg = dg[dg.ra.notna()]
            dg = dg[dg.dec.notna()]
            dg = dg[dg.medval.notna()]
            dg = dg[dg.medval > 0]

            mask = dg.neg1sig<1e-5
            dg.neg1sig[mask] = dg.medval[mask]*0.5
            dg.neg2sig[mask] = 0.0

            mask = dg.pos2sig <= dg.pos1sig
            dg.pos2sig[mask] = dg.pos1sig[mask]*1.1

            dg.loc[:,"theta"] = np.pi/2 - (dg.dec*np.pi/180)
            dg.loc[:,"phi"]   = dg.ra*np.pi/180
            dg.loc[:,"z"]   = dg.medval
            dg.loc[:,"z_err"]   = (dg.pos1sig - dg.neg1sig)*0.5

            dg.loc[:, "z_lowerbound"] = dg.neg2sig
            dg.loc[:, "z_lower"] = dg.neg1sig
            dg.loc[:, "z_upper"] = dg.pos1sig
            dg.loc[:, "z_upperbound"] = dg.pos2sig
            
            dg.loc[:, "w"] = np.ones(dg.shape[0])

            dg = dg[dg.z_err.notna()]

            dg = dg.loc[:,['theta','phi', 'z', 'z_err', 'z_lowerbound', 'z_lower', 'z_upper', 'z_upperbound', 'w']]

            dg.loc[:,"pix" + str(self._nside)]   = hp.ang2pix(self._nside, dg.theta, dg.phi)

            self.data = self.data.append(dg, ignore_index=True)
         
            
            

