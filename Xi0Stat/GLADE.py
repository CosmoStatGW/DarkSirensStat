import pandas as pd
import healpy as hp
import numpy as np

import os, os.path


####
# This module contains a class to handle the GLADE catalogue
####

from Xi0Stat.globals import *
from Xi0Stat.galCat import GalCat


class GLADE(GalCat):
    
    def __init__(self, foldername, compl, subsurveysIncl = ['GWGC', 'HYPERLEDA', 'TWOMASS', 'SDSS'], subsurveysExcl = [], **kwargs):
        print('Initializing GLADE...')
        
        self._subsurveysIncl = subsurveysIncl
        self._subsurveysExcl = subsurveysExcl
        
        assert(set(subsurveysExcl).isdisjoint(subsurveysIncl))
        assert(len(subsurveysIncl) > 0)
               
        GalCat.__init__(self, foldername, compl, **kwargs)
        
    
    def load(self, B_band_select=False,
                   K_band_select=False,
                   z_flag=None,
                   drop_z_uncorr=False,
                   get_cosmo_z=True, #cosmo=None, 
                   pos_z_cosmo=True,
                   drop_no_dist=False,
                   group_correct=True,
                   which_z_correct = 'z_cosmo',
                   group_cat_path = '/Users/Michi/Dropbox/statistical_method_schutz_data/data/Galaxy_Group_Catalogue.csv',
                   CMB_correct=True, l_CMB=263.99, b_CMB=48.26, v_CMB=369,
                   add_B_lum=True, MBSun=5.498,
                   add_K_lum=True, MKSun=3.27,
                   which_z='z_cosmo_CMB',
                   err_vals='GLADE',
                   drop_HyperLeda2=True, 
                   colnames_final = ['theta','phi','z','z_err', 'z_lower', 'z_lowerbound', 'z_upper', 'z_upperbound', 'B_Lum','K_Lum']):
        
        
        gname='GLADE_2.4.txt'
        print('Loading GLADE FROM %s...' %self._path+gname)
        filepath = os.path.join(self._path, gname)

        from astropy.cosmology import FlatLambdaCDM
        cosmoGLADE = FlatLambdaCDM(H0=70.0, Om0=0.27)  # the values used by GLADE

        
        
        # ------ LOAD CATALOGUE
        
        df = pd.read_csv(filepath, sep=" ", header=None, low_memory=False)

        colnames = ['PGC', 'GWGC_name', 'HYPERLEDA_name', 'TWOMASS_name', 'SDSS_name', 'flag1', 'RA', 'dec',
                    'dL', 'dL_err', 'z', 'B', 'B_err', 'B_Abs', 'J', 'J_err', 'H', 'H_err', 'K', 'K_err',
                    'flag2', 'flag3'
                   ]
        
        df.columns=colnames
        
         
        # ------  SELECT SUBSURVEYS
        
        
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
        
        
        # ------ Add theta, phi for healpix
        
        df.loc[:,"theta"] = np.pi/2 - (df.dec*np.pi/180)
        df.loc[:,"phi"]   = df.RA*np.pi/180
  
        
        
        
        or_dim = df.shape[0] # ORIGINAL LENGTH OF THE CATALOGUE
        print('N. of objects: %s' %or_dim)
        
       
            
        
        # ------ Select parts of the catalogue
        
        if B_band_select and K_band_select:
            print('Keeping only galaxies with known apparent magnitude in B and K band...')
            df=df[(df.B.notna()==True) &(df.K.notna()==True)] #select the galaxies that have a measured value of B luminosity
            print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
            
        
        elif B_band_select:
            print('Keeping only galaxies with known apparent magnitude in B band...')
            df=df[df.B.notna()==True] #select the galaxies that have a measured value of B luminosity
            print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
        
        elif K_band_select:
            print('Keeping only galaxies with known apparent magnitude in K band...')
            df=df[df.K.notna()==True] #select the galaxies that have a measured value of B luminosity
            print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
        
        if z_flag is not None:
            print('Dropping galaxies with flag2=%s...' %z_flag)
            df=  df[df.flag2 != z_flag ]
            print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
            
        
        if drop_z_uncorr:
            print('Keeping only galaxies with redshift corrected for peculiar velocities...')
            df=df[df['flag3']==1]
            print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
          
            
        if drop_no_dist:
            print('Keeping only galaxies with known value of luminosity distance...')
            df=df[df.dL.notna()==True]
            print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
        
        if drop_HyperLeda2:
            print("Dropping galaxies with HyperLeda name=null and flag2=2...")
            df=df.drop(df[(df['HYPERLEDA_name'].isna()) & (df['flag2']==2)].index)
            print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.00%}".format(df.shape[0]/or_dim)+' of total' )
        
        
        
        # ------ Add z corrections
        
        
        if get_cosmo_z:
            print('Computing cosmological redshifts from given luminosity distance with H0=%s, Om0=%s...' %(cosmoGLADE.H0, cosmoGLADE.Om0))

            
            z_max = df[df.dL.notna()]['z'].max() +0.01
            z_min = max(0, df[df.dL.notna()]['z'].min() - 1e-05)
            print('Interpolating between z_min=%s, z_max=%s' %(z_min, z_max))
            z_grid = np.linspace(z_min, z_max, 200000)
            dL_grid = cosmoGLADE.luminosity_distance(z_grid).value
            
            if not drop_no_dist:
                dLvals = df[df.dL.notna()]['dL']
                print('%s points have valid entry for dist' %dLvals.shape[0])
                zvals = df[df.dL.isna()]['z']
                print('%s points have null entry for dist, correcting original redshift' %zvals.shape[0])
                z_cosmo_vals = np.where(df.dL.notna(), np.interp( df.dL , dL_grid, z_grid), df.z)             
            else:
                z_cosmo_vals = np.interp( df.dL , dL_grid, z_grid)
            
            df.loc[:,'z_cosmo'] = z_cosmo_vals
                
            
            
            
            if not CMB_correct and not group_correct and pos_z_cosmo:
                print('Keeping only galaxies with positive cosmological redshift...')
                df = df[df.z_cosmo >= 0]
                print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
        
        
        if group_correct:
            if not get_cosmo_z:
                raise ValueError('To apply group corrections, compute cosmological redshift first')
            df_groups =  pd.read_csv(group_cat_path)
            df = self.group_correction(df, df_groups, which_z=which_z_correct)
            
      
        
        if CMB_correct:
            if not get_cosmo_z:
                raise ValueError('To apply CMB corrections, compute cosmological redshift first')

            df = self.CMB_correction(df, which_z=which_z_correct)
            if pos_z_cosmo:
                print('Keeping only galaxies with positive redshift in the colums %s...' %which_z)
                df = df[df[which_z ]>= 0]
                print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
        
        if which_z!='z':
            print('Renaming column %s to z. This will be used in the analysis.' %which_z)
            df = df.drop(columns='z')
            df.rename(columns={which_z:'z'}, inplace=True)
        # From now on, the relevant column for redshift, including all corrections, will be 'z'
        
        # ------ Add z errors
        
        if err_vals is not None:
            print('Adding errors on z with %s values' %err_vals)
            if err_vals=='GLADE':
                scales = np.where(df['flag2'].values==3, 1.5*1e-04, 1.5*1e-02)
            elif err_vals=='const_perc':
                scales=np.where(df['flag2'].values==3, df.z/100, df.z/10)
            elif err_vals=='const':
                scales = np.full(df.shape[0], 200/clight) 
            else:
                raise ValueError('Enter valid choice for err_vals. Valid options are: GLADE, const, const_perc . Got %s' %err_vals)
            
            df.loc[:, 'z_err'] = scales
        
            df.loc[:, 'z_lowerbound'] = df.z - 3*df.z_err
            df.loc[df.z_lowerbound < 0, 'z_lowerbound'] = 0
            df.loc[:, 'z_lower'] = df.z - df.z_err
            df.loc[df.z_lower < 0.5*df.z, 'z_lower'] = 0.5*df.z
            df.loc[:, 'z_upper'] = df.z + df.z_err
            df.loc[:, 'z_upperbound'] = df.z + 3*df.z_err
        
        # ------ Add B luminosity
        
        if add_B_lum:
            print('Computing total luminosity in B band...')
            # add  a column for B-band luminosity 
            #my_dist=cosmo.luminosity_distance(df.z.values).value
            #df.loc[:,"B_Abs"]=df.B-5*np.log10(my_dist)-25
            BLum = df.B_Abs.apply(lambda x: TotLum(x, MBSun))
            df.loc[:,"B_Lum"] =BLum
            df = df.drop(columns='B_Abs')
            df = df.drop(columns='B')
            #print('Done.')
        
        
        # ------ Add K luminosity
        
        if add_K_lum:
            print('Computing total luminosity in K band...')
            my_dist=cosmoGLADE.luminosity_distance(df.z.values).value
            df.loc[:,"K_Abs"]=df.K-5*np.log10(my_dist)-25
            KLum = df.K_Abs.apply(lambda x: TotLum(x, MKSun))
            df.loc[:,"K_Lum"]=KLum
            df = df.drop(columns='K_Abs')
            df = df.drop(columns='K')
        
        
    
        #dg=df.loc[:,['theta','phi','z','z_err', 'z_lower', 'z_lowerbound', 'z_upper', 'z_upperbound', 'B_Lum','K_Lum']]
        
        df.loc[:, 'w'] = 1 #dg.B_Lum_over_h7sq
        df.loc[:,"pix"  + str(self._nside)]   = hp.ang2pix(self._nside, df.theta, df.phi)
        
        # ------ Keep only given columns
        
        if colnames_final is not None:
            print('Keeping only columns: %s' %colnames_final)
            df = df[colnames_final]
        
        # ------ 
        print('Done.')
        
        
        self.data = self.data.append(df, ignore_index=True)
        
      

   
def TotLum(x, MSun): 
    return 10**(-0.4* (x+25-MSun))