import pandas as pd
import healpy as hp
import numpy as np

import os, os.path


####
# This module contains a class to handle the GLADE catalogue
####

from globals import *
from galCat import GalCat



class GLADE(GalCat):
    
    def __init__(self, foldername, compl, useDirac, #finalData = None,
                 subsurveysIncl = ['GWGC', 'HYPERLEDA', 'TWOMASS', 'SDSS'], 
                 subsurveysExcl = [], 
                 verbose=False,
                 **kwargs):
        
        self._subsurveysIncl = subsurveysIncl
        self._subsurveysExcl = subsurveysExcl
        #self._finalData = finalData
        
        assert(set(subsurveysExcl).isdisjoint(subsurveysIncl))
        assert(len(subsurveysIncl) > 0)

        GalCat.__init__(self, foldername, compl, useDirac, verbose, **kwargs)
        
    
    def load(self, band=None, band_weight=None,
                   Lcut=0,
                   zMax = 100,
                   z_flag=None,
                   drop_z_uncorr=False,
                   get_cosmo_z=True, #cosmo=None, 
                   pos_z_cosmo=True,
                   drop_no_dist=False,
                   group_correct=True,
                   which_z_correct = 'z_cosmo',
                   CMB_correct=True,
                   which_z='z_cosmo_CMB',
                   galPosterior = True,
                   err_vals='GLADE',
                   drop_HyperLeda2=True, 
                   colnames_final = ['theta','phi','z','z_err', 'z_lower', 'z_lowerbound', 'z_upper', 'z_upperbound', 'w']):
        
        if band_weight is not None:
            assert band_weight==band
       
        loaded = False
        computePosterior = False
        posteriorglade = os.path.join(self._path, 'posteriorglade.csv')
        if self.verbose:
            print(posteriorglade)
        if galPosterior:
            from os.path import isfile
            if isfile(posteriorglade):
                if self.verbose:
                    print("Directly loading final data ")
                df = pd.read_csv(os.path.join(self._path, 'posteriorglade.csv'))
                loaded=True
            else:
                computePosterior=True
                loaded=False
                
            
        if not loaded:
        
            
            gname='GLADE_2.4.txt'
            groupname='Galaxy_Group_Catalogue.csv'
            filepath_GLADE = os.path.join(self._path, gname)
            filepath_groups = os.path.join(miscPath, groupname)

            from astropy.cosmology import FlatLambdaCDM
            cosmoGLADE = FlatLambdaCDM(H0=H0GLADE, Om0=Om0GLADE)  # the values used by GLADE

            
            
            # ------ LOAD CATALOGUE
            if self.verbose:
                print('Loading GLADE from %s...' %filepath_GLADE)
            df = pd.read_csv(filepath_GLADE, sep=" ", header=None, low_memory=False)

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
            
            
            # ------ Add theta, phi for healpix in radians
            
            df.loc[:,"theta"] = np.pi/2 - (df.dec*np.pi/180)
            df.loc[:,"phi"]   = df.RA*np.pi/180
      
            
            
            
            or_dim = df.shape[0] # ORIGINAL LENGTH OF THE CATALOGUE
            if self.verbose:
                print('N. of objects: %s' %or_dim)
            
           
                
            
            # ------ Select parts of the catalogue
                    
            if z_flag is not None:
                df=  df[df.flag2 != z_flag ]
                if self.verbose:
                    print('Dropping galaxies with flag2=%s...' %z_flag)
                    print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
                
            
            if drop_z_uncorr:
                df=df[df['flag3']==1]
                if self.verbose:
                    print('Keeping only galaxies with redshift corrected for peculiar velocities...')
                
                    print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
              
                
            if drop_no_dist:
                df=df[df.dL.notna()==True]
                if self.verbose:
                    print('Keeping only galaxies with known value of luminosity distance...')
                    print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
            
            if drop_HyperLeda2:
                df=df.drop(df[(df['HYPERLEDA_name'].isna()) & (df['flag2']==2)].index)
                if self.verbose:
                    print("Dropping galaxies with HyperLeda name=null and flag2=2...")
                    print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.00%}".format(df.shape[0]/or_dim)+' of total' )
            
            
            
            # ------ Add z corrections
            
            
            if get_cosmo_z:
                if self.verbose:
                    print('Computing cosmological redshifts from given luminosity distance with H0=%s, Om0=%s...' %(cosmoGLADE.H0, cosmoGLADE.Om0))

                
                z_max = df[df.dL.notna()]['z'].max() +0.01
                z_min = max(0, df[df.dL.notna()]['z'].min() - 1e-05)
                if self.verbose:
                    print('Interpolating between z_min=%s, z_max=%s' %(z_min, z_max))
                z_grid = np.linspace(z_min, z_max, 200000)
                dL_grid = cosmoGLADE.luminosity_distance(z_grid).value
                
                if not drop_no_dist:
                    dLvals = df[df.dL.notna()]['dL']
                    if self.verbose:
                        print('%s points have valid entry for dist' %dLvals.shape[0])
                    zvals = df[df.dL.isna()]['z']
                    if self.verbose:
                        print('%s points have null entry for dist, correcting original redshift' %zvals.shape[0])
                    z_cosmo_vals = np.where(df.dL.notna(), np.interp( df.dL , dL_grid, z_grid), df.z)
                else:
                    z_cosmo_vals = np.interp( df.dL , dL_grid, z_grid)
                
                df.loc[:,'z_cosmo'] = z_cosmo_vals
                    
                
                
                
                if not CMB_correct and not group_correct and pos_z_cosmo:
                    if self.verbose:
                        print('Keeping only galaxies with positive cosmological redshift...')
                    df = df[df.z_cosmo >= 0]
                    if self.verbose:
                        print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
            
            
            if group_correct:
                if not get_cosmo_z:
                    raise ValueError('To apply group corrections, compute cosmological redshift first')
                if self.verbose:
                    print('Loading galaxy group catalogue from %s...' %filepath_groups)
                df_groups =  pd.read_csv(filepath_groups)
                self.group_correction(df, df_groups, which_z=which_z_correct)
                
          
            
            if CMB_correct:
                if not get_cosmo_z:
                    raise ValueError('To apply CMB corrections, compute cosmological redshift first')

                self.CMB_correction(df, which_z=which_z_correct)
                if pos_z_cosmo:
                    if self.verbose:
                        print('Keeping only galaxies with positive redshift in the colums %s...' %which_z)
                    df = df[df[which_z ]>= 0]
                    if self.verbose:
                        print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
            
            if which_z!='z':
                if self.verbose:
                    print('Renaming column %s to z. This will be used in the analysis.' %which_z)
                df = df.drop(columns='z')
                df.rename(columns={which_z:'z'}, inplace=True)
            # From now on, the relevant column for redshift, including all corrections, will be 'z'
            
            # ------ Potentially drop large z
            
            df = df[df.z < zMax]
            
            
            # ------ Add z errors
            
            if err_vals is not None:
                if self.verbose:
                    print('Adding errors on z with %s values' %err_vals)
                if err_vals=='GLADE':
                    scales = np.where(df['flag2'].values==3, 1.5*1e-04, 1.5*1e-02)
                elif err_vals=='const_perc':
                    scales=np.where(df['flag2'].values==3, df.z/100, df.z/10)
                elif err_vals=='const':
                    scales = np.full(df.shape[0], 200/clight)
                else:
                    raise ValueError('Enter valid choice for err_vals. Valid options are: GLADE, const, const_perc . Got %s' %err_vals)
                
                # restrict error to <=z itself. otherwise for z very close to 0 input is infeasible for keelin distributions, which would break things silently
                df.loc[:, 'z_err'] = np.minimum(scales, df.z.to_numpy())
            
                df.loc[:, 'z_lowerbound'] = df.z - 3*df.z_err
                df.loc[df.z_lowerbound < 0, 'z_lowerbound'] = 0
                df.loc[:, 'z_lower'] = df.z - df.z_err
                df.loc[df.z_lower < 0.5*df.z, 'z_lower'] = 0.5*df.z
                df.loc[:, 'z_upper'] = df.z + df.z_err
                df.loc[:, 'z_upperbound'] = df.z + 3*df.z_err
            
                # ------ Estimate galaxy posteriors with contant-in-comoving prior
                
                if computePosterior:
                    
                    self.include_vol_prior(df)
#                    L = 0
#                    nBatch = 10000
#
#                    if self.verbose:
#                        print("Computing galaxy posteriors...")
#
#                    from keelin import convolve_bounded_keelin_3
#                    from astropy.cosmology import FlatLambdaCDM
#                    fiducialcosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
#                    zGrid = np.linspace(0, 2*np.max(df.z_upperbound), 500)
#                    jac = fiducialcosmo.comoving_distance(zGrid).value**2 / fiducialcosmo.H(zGrid).value
#
#                    from scipy import interpolate
#                    func = interpolate.interp1d(zGrid, jac, kind='cubic')
#
#                    i = 0
#                    while True:
#                        i += 1
#                        if self.verbose:
#                            print("Batch " + str(i) + " of " + str(np.int(len(df)/nBatch)+1) )
#
#                        R = L + nBatch
#
#                        if R >= len(df):
#                            ll = df.z_lowerbound.to_numpy()[L:]
#                            l  = df.z_lower.to_numpy()[L:]
#                            m  = df.z.to_numpy()[L:]
#                            u  = df.z_upper.to_numpy()[L:]
#                            uu = df.z_upperbound.to_numpy()[L:]
#                        else:
#                            ll = df.z_lowerbound.to_numpy()[L:R]
#                            l  = df.z_lower.to_numpy()[L:R]
#                            m  = df.z.to_numpy()[L:R]
#                            u  = df.z_upper.to_numpy()[L:R]
#                            uu = df.z_upperbound.to_numpy()[L:R]
#
#
#                        fits = convolve_bounded_keelin_3(func, 0.16, l, m, u, ll, uu, N=500)
#
#                        if R >= len(df):
#                            df.iloc[L:, df.columns.get_loc("z_lowerbound")] = fits[:, 0]
#                            df.iloc[L:, df.columns.get_loc("z_lower")] = fits[:, 1]
#                            df.iloc[L:, df.columns.get_loc("z")] = fits[:, 2]
#                            df.iloc[L:, df.columns.get_loc("z_upper")] = fits[:, 3]
#                            df.iloc[L:, df.columns.get_loc("z_upperbound")] = fits[:, 4]
#                            break
#                        else:
#                            df.iloc[L:R, df.columns.get_loc("z_lowerbound")] = fits[:, 0]
#                            df.iloc[L:R, df.columns.get_loc("z_lower")] = fits[:, 1]
#                            df.iloc[L:R, df.columns.get_loc("z")] = fits[:, 2]
#                            df.iloc[L:R, df.columns.get_loc("z_upper")] = fits[:, 3]
#                            df.iloc[L:R, df.columns.get_loc("z_upperbound")] = fits[:, 4]
#
#
#                        L += nBatch
#
        # ------ End if not use precomputed table
        #        Always be able to still chose the weighting and cut.
        
        if band=='B' or band_weight=='B':
            add_B_lum=True
            add_K_lum=False
        elif band=='K' or band_weight=='B':
            add_B_lum=False
            add_K_lum=True
        else:
            add_B_lum=False
            add_K_lum=False
            
            
        # ------ Add B luminosity
        
        if add_B_lum:
            if self.verbose:
                print('Computing total luminosity in B band...')
            # add  a column for B-band luminosity
            #my_dist=cosmo.luminosity_distance(df.z.values).value
            #df.loc[:,"B_Abs"]=df.B-5*np.log10(my_dist)-25
            BLum = df.B_Abs.apply(lambda x: TotLum(x, MBSun))
            df.loc[:,"B_Lum"] =BLum
            df = df.drop(columns='B_Abs')
            # df = df.drop(columns='B') don't assume it's here
            #print('Done.')
        
        
        # ------ Add K luminosity
        
        if add_K_lum:
            if self.verbose:
                print('Computing total luminosity in K band...')
            my_dist=cosmoGLADE.luminosity_distance(df.z.values).value
            df.loc[:,"K_Abs"]=df.K-5*np.log10(my_dist)-25
            KLum = df.K_Abs.apply(lambda x: TotLum(x, MKSun))
            df.loc[:,"K_Lum"]=KLum
            # df = df.drop(columns='K_Abs') don't assume it's here
            df = df.drop(columns='K')
        
        
        # ------ Apply cut in luminosity
        if band is not None:
            col_name=band+'_Lum'
            L_th = Lcut*LBstar07
            if self.verbose:
                print('Applying cut in luminosity in %s-band. Selecting galaxies with %s>%s x L_* = %s' %(band, col_name, Lcut, np.round(L_th,2)))
            or_dim = df.shape[0]
            df = df[df[col_name]>L_th]
            if self.verbose:
                print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
                       
        else:
            if self.verbose:
                print('No cut in luminosity applied ' )
            #w = np.ones(df.shape[0])
        
         
        # ------ Add 'w' column for weights
        if band_weight is not None:
            w_name=band+'_Lum'
            w = df.loc[:, w_name].values
            if self.verbose:
                print('Using %s for weighting' %col_name)
        else:
            w = np.ones(df.shape[0])
            if self.verbose:
                print('Using weights =1 .')
            
        df.loc[:, 'w'] = w
        
        
        # ------ Keep only some columns
        
        if colnames_final is not None:
            if self.verbose:
                print('Keeping only columns: %s' %colnames_final)
            df = df[colnames_final]
       
        # ------ Add pixel column. Note that not providing nest parameter to ang2pix defaults to nest=True, which has to be set in GW too!
        df.loc[:,"pix"  + str(self._nside)]   = hp.ang2pix(self._nside, df.theta, df.phi)
        
        # ------
        if self.verbose:
            print('GLADE loaded.')
        
        
        self.data = self.data.append(df, ignore_index=True)
            
      

   
def TotLum(x, MSun): 
    return 10**(-0.4* (x+25-MSun))
