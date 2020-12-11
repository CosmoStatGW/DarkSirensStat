
####
# This module contains a class to handle the GWENS catalogue
####


import pandas as pd
import healpy as hp
import numpy as np

import os, os.path

from galCat import GalCat
from globals import *

class GWENS(GalCat):
    
    def __init__(self, foldername, compl, useDirac, patches=[], verbose = False, **kwargs):
        print('Loading GWENS...')
       
        self._patches = patches
            
        GalCat.__init__(self, foldername, compl, useDirac, verbose, **kwargs)
    
    def load(self, galPosterior = True):

        if self._patches == []:
            filenames = [os.path.join(self._path, name) for name in os.listdir(self._path) if (os.path.isfile(os.path.join(self._path, name)) and '.csv.gz' in name) ]
        else:
            filenames = [os.path.join(self._path, 'ra_{:03d}_{:03d}.csv.gz'.format(i*15, (i+1)*15)) for i in self._patches]

        for i, filename in enumerate(filenames):
        
            print('Loading patch ' + str(i) + ' of ' + str(len(filenames)) + ' patches of GWENS...')
            
            dg = pd.read_csv(filename, compression='gzip', error_bad_lines=False)
            #dg = pd.read_csv(filename, error_bad_lines=False)
           
            if len(dg) < 1000:
                continue

#            dg = dg[dg.cmodelMag_r < 21.8]
            dg = dg.loc[:,['ra','dec', 'medval', 'neg2sig', 'neg1sig', 'pos1sig', 'pos2sig', 'specflag', 'lstar', 'cmodelMag_r', 'extinction_r']]

            dg = dg[dg.ra.notna()]
            dg = dg[dg.dec.notna()]
            dg = dg[dg.medval.notna()]
            dg = dg[dg.neg1sig.notna()]
            dg = dg[dg.neg2sig.notna()]
            dg = dg[dg.pos1sig.notna()]
            dg = dg[dg.pos2sig.notna()]

            dg = dg[dg.pos2sig < 6]

            dg = dg[dg.medval > 0]
            dg = dg[dg.medval > dg.neg1sig]
            dg = dg[dg.pos1sig > dg.medval]

            # one sigma down 
            dg.loc[:, 'z_lowerbound'] = dg.neg2sig - (dg.medval - dg.neg1sig)

            mask = dg.z_lowerbound < 0.0
            dg.loc[mask, 'z_lowerbound'] = 0.0

            # remove if still too compressed 

            dg = dg[dg.z_lowerbound < dg.medval - 2*(dg.medval - dg.neg1sig)]

            # close to zero?
            mask = dg.neg1sig < 1e-5
            dg.loc[mask, 'neg1sig'] = dg.medval[mask]*0.5
            dg.loc[mask, 'z_lowerbound'] = 0.0

            dg.loc[:,"z"]   = dg.medval
            dg.loc[:,"z_err"]   = (dg.pos1sig - dg.neg1sig)*0.5

            # drop useless galaxies
            dg = dg[dg.z_err < 1]
            # drop weird pdfs
            dg = dg[dg.z_err < 3*dg.z] 
           
            # many galaxies have infeasible keelin distributions due to too low pos2sig 
            mask = dg.pos2sig <= dg.pos1sig + 2*(dg.pos1sig-dg.z)
            dg.loc[mask, "pos2sig"] = dg.pos1sig[mask] + 3*(dg.pos1sig[mask]-dg.z[mask])

            #mask = (dg.pos1sig <= dg.z) | (dg.z <= dg.neg1sig) | (dg.neg1sig <= dg.neg2sig) 
            #dg = dg[~mask]

            dg.loc[:, "z_lower"] = dg.neg1sig
            dg.loc[:, "z_upper"] = dg.pos1sig
            dg.loc[:, "z_upperbound"] = dg.pos2sig
            
            
            dg.loc[:, "w"] = np.ones(dg.shape[0])
            dg.loc[:,"theta"] = np.pi/2 - (dg.dec*np.pi/180)
            dg.loc[:,"phi"]   = dg.ra*np.pi/180

            dg = dg[dg.z_err.notna()]

            if galPosterior:
                dg = self.include_vol_prior(dg) 

            dLmax = dL70fast(0.5)
            rappCut = 21.8
            typicalrExtinction = 0.5
            rabsCut = rappCut - typicalrExtinction - 5*np.log10(dLmax) - 25

            dg.loc[:, 'rabs'] = dg.cmodelMag_r - dg.extinction_r - 5*np.log10(dL70fast(dg.z.to_numpy())) - 25

            before = len(dg)
            dg = dg[dg.rabs < rabsCut] 
            after = len(dg)
            print("Dropped {:.2} percent of the galaxies due to r band cut.".format((before-after)/before*100))

            dg = dg.loc[:,['theta','phi', 'z', 'z_err', 'z_lowerbound', 'z_lower', 'z_upper', 'z_upperbound', 'w']]
            dg.loc[:,"pix" + str(self._nside)]   = hp.ang2pix(self._nside, dg.theta, dg.phi)

            self.data = self.data.append(dg, ignore_index=True)

        
            
            
#        if galPosterior:
#            self.data = self.include_vol_prior(self.data) 

