import pandas as pd
import healpy as hp
import numpy as np

import os, os.path


####
# This module contains a class to handle the GLADE catalogue
####

from Xi0Stat.galCat import GalCat


class GLADE(GalCat):
    
    def __init__(self, foldername, compl, subsurveysIncl = ['GWGC', 'HYPERLEDA', 'TWOMASS'], subsurveysExcl = ['SDSS'], **kwargs):
        print('Initializing GLADE...')
        
        self._subsurveysIncl = subsurveysIncl
        self._subsurveysExcl = subsurveysExcl
        
        assert(set(subsurveysExcl).isdisjoint(subsurveysIncl))
        assert(len(subsurveysIncl) > 0)
               
        GalCat.__init__(self, foldername, compl, **kwargs)
        
    
    def load(self):
        filepath = os.path.join(self._path, 'GLADE_2.4.txt')

        from astropy.cosmology import FlatLambdaCDM
        cosmoGLADE = FlatLambdaCDM(H0=70.0, Om0=0.27)  # the values used by GLADE

        drop_gals_without_dl = False
        CMB_CORR = True

        df = pd.read_csv(filepath, sep=" ", header=None, low_memory=False)

        colnames = ['PGC', 'GWGC_name', 'HYPERLEDA_name', 'TWOMASS_name', 'SDSS_name', 'flag1', 'RA', 'dec',
                    'dL', 'dL_err', 'z', 'B', 'B_err', 'B_Abs', 'J', 'J_err', 'H', 'H_err', 'K', 'K_err',
                    'flag2', 'flag3'
                   ]
        df.columns=colnames

        # if object is named in a survey (not NaN), it is present in a survey
        for survey in ['GWGC', 'HYPERLEDA', 'TWOMASS', 'SDSS']:
            # new column named suvey that will be True or False
            # if object is in or not
            
            #copy the name column
            df.loc[:,survey] = df[survey + '_name']
            
            # NaN to False
            df[survey] = df[survey].fillna(False)
            # not False (i.e. a name) to True
            df.loc[df[survey] != False, survey] = True

      
        # include only objects that are contained in at least one of the surveys listed in _subsurveysIncl (is non-empty list)
        mask = df[self._subsurveysIncl[0]] == True
        for incl in self._subsurveysIncl[1:]:
            mask = mask | (df[incl] == True)
        df = df.loc[mask]
        
        # explicitely exclude objects if they are contained in some survey(s) in _subsurveysExcl
        # (may be necessary if an object is in multiple surveys)
        for excl in self._subsurveysExcl:
            df = df.loc[df[excl] == False]

        df.loc[:,"theta"] = np.pi/2 - (df.dec*np.pi/180)
        df.loc[:,"phi"]   = df.RA*np.pi/180
  

        MBSun=5.498
        MKSun=3.27
        absmag2lum = lambda x, M: 10**(-0.4* (x-M+25)) #25 to get 10^10 L(M) as units, which is nice when L is L_sun (so M should be M_sun)
        appmag2lum = lambda x, M, dL: 10**(-0.4* (x-M))*dL**2
        df.loc[:,"B_Lum_over_h7sq"] = absmag2lum(df.B_Abs.values, MBSun)
        df.loc[:,"K_Lum_over_h7sq"] = appmag2lum(df.K.values, MKSun, df.dL.values)

        #### REDSHIFTS ####

        maxz = np.max(df.z)
    
        # interpolation table for fast inversion
        z_table_GLADE = np.linspace(0, maxz, 3000)
        dL_table_GLADE = cosmoGLADE.luminosity_distance(z_table_GLADE).value

        from scipy import interpolate
        redsh = interpolate.interp1d(dL_table_GLADE, z_table_GLADE, kind='cubic')
        
        # dL is partly corrected for some sources vs. z, so use that. note that it has not been corrected for earth's motion wrt to CMB - doppler shifts, to be done below on z
        df.loc[df.dL.notna(), 'z'] = redsh(df.dL[df.dL.notna()])

        if drop_gals_without_dl:
            print("Dropping all galaxies without luminosity distance... All remaining redshifts automatically positive")
            df = df[df.dL.notna()]
            
        #### CMB CORRECTION

        if CMB_CORR:
        #    print("Correcting redshifts to CMB frame")
            l_CMB = 263.99 *np.pi/180
            b_CMB = 48.26*np.pi/180

            clight= 2.99792458* 10**5 #speed of light in km/s
            v_CMB = 369.0 # in km/s

            z_CMB = v_CMB/clight

            rot = hp.Rotator(coord=['G','C'])
            cmb = rot(np.pi*0.5 - b_CMB, l_CMB)

            df.z += z_CMB*(np.cos(df.theta)*np.cos(cmb[0]) + np.sin(df.theta)*np.sin(cmb[0])*np.cos(df.phi - cmb[1]))


        df.loc[:, 'z_err'] = 1e-2 #-2 for photo, -4 for spectro  z = 0.005

        df.loc[:, 'z_lowerbound'] = df.z - 3*df.z_err
        df.loc[df.z_lowerbound < 0, 'z_lowerbound'] = 0
        df.loc[:, 'z_lower'] = df.z - df.z_err
        df.loc[df.z_lower < 0.5*df.z, 'z_lower'] = 0.5*df.z
        df.loc[:, 'z_upper'] = df.z + df.z_err
        df.loc[:, 'z_upperbound'] = df.z + 3*df.z_err

        print("Number of remaining galaxies with negative redshift is {}. Dropping them...".format(np.sum(df.z<0)))
        df = df[df.z >= 0]

        dg=df.loc[:,['theta','phi','z','z_err', 'z_lower', 'z_lowerbound', 'z_upper', 'z_upperbound', 'B_Lum_over_h7sq','K_Lum_over_h7sq']]
        
        dg.loc[:, 'w'] = 1 #dg.B_Lum_over_h7sq
        dg.loc[:,"pix"  + str(self._nside)]   = hp.ang2pix(self._nside, dg.theta, dg.phi)
        self.data = self.data.append(dg, ignore_index=True)
        
      

   
