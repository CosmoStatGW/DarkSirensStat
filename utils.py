#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:10:48 2020

@author: Michi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sys

PRIOR_LOW = 0.2
PRIOR_UP = 3
PRIOR_UP_H0 = 220
PRIOR_LOW_H0 = 10

clight= 2.99792458* 10**5

l_CMB, b_CMB = (263.99, 48.26)
v_CMB = 369

MBsun_val=5.498

H0_GLADE=70
Om0_GLADE=0.27

# column names for GLADE
colnames_GLADE = ['PGC', 'GWGC name', 'HyperLEDA_name', '2MASS_name', 'SDSS-DR12_name', 'flag1', 'RA', 'dec', 
            'dist', 'dist_err', 'z', 'B', 'B_err', 'B_Abs', 'J', 'J_err', 'H', 'H_err', 'K', 'K_err',
            'flag2', 'flag3']

# sublist of columns to keep
col_list_GLADE=  ['PGC', 'GWGC name','HyperLEDA_name','RA', 'dec', 'dist', 'z',  'B', 'K', 'flag2']








def edit_catalogue(df,
                   col_list=None,
                   B_band_select=False,
                   K_band_select=True,
                   z_flag=None,
                   drop_z_uncorr=False,
                   get_cosmo_z=True, cosmo=None, 
                   pos_z_cosmo=True,
                   drop_no_dist=True,
                   group_correct=False,
                   which_z_correct = 'z_cosmo',
                   group_cat_path = '/Users/Michi/Dropbox/statistical_method_schutz_data/data/Galaxy_Group_Catalogue.csv',
                   CMB_correct=True, l_CMB=263.99, b_CMB=48.26, v_CMB=369,
                   add_B_lum=True, MBSun=5.498,
                   add_K_lum=True, MKSun=3.27,
                   which_z='z_cosmo_corr',
                   err_vals='GLADE',
                   drop_HyperLeda2=True,
                   ):
    
        '''
        
        Input: df. DataFrame containing GLADE catalogue.
                    It is assumed that it has columns with following names:
                    ['PGC', 'GWGC name', 'HyperLEDA_name', '2MASS_name', 'SDSS-DR12_name', 'flag1', 'RA', 'dec', 
                     'dist', 'dist_err', 'z', 'B', 'B_err', 'B_Abs', 'J', 'J_err', 'H', 'H_err', 'K', 'K_err',
                     'flag2', 'flag3'
                     ]
                    - see http://glade.elte.hu/Download.html for explanation
        
        Options:
            - col_list: list of strings. Contains names of the columns that have to be kept
            - B_band_select : boolean. If true, keep galaxies with measured apparent B magnitude 
            - K_band_select : boolean. If true, keep galaxies with measured apparent K magnitude  
            - z_flag : integer . If specified, only galaxies with corresponding value for flag2 will be kept 
                        (possible values: 0, 1, 2, 3) 
            - drop_z_uncorr: Boolean. If true, keep only galaxies for which peculiar velocity corrections have been applied
                        (i.e. flag3==1)
            - get_cosmo_z: boolean. If true, computes the cosmological redshift starting from the GLADE luminosity distance
                                    with the cosmomogy given by cosmo
            - cosmo :  astropy.cosmology.FlatLambdaCDM object. Cosmology used to convert from luminosity distance to redshift 
            - drop_no_dist: boolean. If true, keep only galaxies with valid value for the luminosity distance
            - group_correct: boolean. If true, correct redshift for group velocities. The redshift to correct is specified by the parameter which_z_correct
                            Result will be supercript on the columns given. The original redshift will be stored in a new column 'z_original'
            - which_z_correct: string. Specifies the name of the df column to apply group and CMB correction.
                                Default: z_cosmo
            - group_cat_path: string. Path where the group velocity catalogue used to apply group corrections is saved
            - CMB_correct: boolean. If true, apply CMB reference frame correction to the redshif specified by which_z_correct
                            (NOTE: if group_correct is also True, the group correction will be applied before the CMB)
                            The output will be saved in a column 'z_corr'
            - l_CMB, b_CMB, v_CMB :  galactic latitude, longitude of the CMB dipole  and CMB dipole velocity in km/s
            - add_B_lum :  boolean. If true, add B luminosity in units of 10^10solar lB-uminosity, starting from apparent B magnitude
            - MBSun: float. Solar luminosity in the B band 
            - add_K_lum :  boolean. If true, add K luminosity in units of 10^10solar K-luminosity, starting from apparent K magnitude
        
        
        Output: df with all requested computations/selections performed
        
        
        Example:
            
            import pandas as pd
            import numpy as np
            from astropy.cosmology import FlatLambdaCDM
            
            H0=70
            Om0=0.27
            l_CMB, b_CMB, v_CMB  = 263.99, 48.26, 369
            
            GLADE_PATH = 'GLADE_2.3.txt'
            
            colnames = ['PGC', 'GWGC name', 'HyperLEDA_name', '2MASS_name', 'SDSS-DR12_name', 'flag1', 'RA', 'dec', 
            'dist', 'dist_err', 'z', 'B', 'B_err', 'B_Abs', 'J', 'J_err', 'H', 'H_err', 'K', 'K_err',
            'flag2', 'flag3']
            
            
            cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
            
            df =  pd.read_csv(GLADE_PATH, sep=' ', header=None)
            df.columns=colnames
            
            
            col_list=  ['PGC', 'GWGC name','RA', 'dec', 'dist', 'z',  'B', 'K']
            
            df = edit_catalogue(df,
                                  col_list=col_list, 
                                  B_band_select=False,
                                  K_band_select=False,
                                  z_flag=None,
                                  drop_z_uncorr=False,
                                  get_cosmo_z=True, cosmo=cosmo,
                                  drop_no_dist=True,
                                  group_correct=False,
                                  which_z_correct = 'z_cosmo',
                                  CMB_correct=True, l_CMB=l_CMB, b_CMB=b_CMB, v_CMB=v_CMB,
                                  add_B_lum=True, MBSun=5.48,
                                  add_K_lum=True, MKSun=3.27)
            
            df.head()
            
            	    GWGC name	RA	dec	dist	z	z_cosmo	z_corr	B_Lum	K_Lum
            0	2789.0	NGC0253	11.888060	-25.288799	3.925951	0.000916	0.000916	-0.000030	2.779008	9.270214
            1	46957.0	NGC5128	201.365646	-43.018711	3.767434	0.000879	0.000879	0.001726	2.249527	7.319667
            2	NaN	NaN	56.702141	68.096107	8.355748	0.001948	0.001948	0.001645	0.003079	15.862464
            3	48082.0	NGC5236	204.253000	-29.865500	4.467321	0.001042	0.001042	0.001970	1.888401	NaN
            4	43495.0	NGC4736	192.721451	41.120152	4.245715	0.000991	0.000991	0.001729	0.847036	3.135463

        
        ''' 
        
        

        
        
        # ------ Keep only given columns
        
        or_dim = df.shape[0] # ORIGINAL LENGTH OF THE CATALOGUE
        print('N. of objects: %s' %or_dim)
        
        if col_list is not None:
            print('Keeping only columns: %s' %col_list)
            df = df[col_list]
        
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
          
        #if get_cosmo_z:
        #    drop_no_dist=True 
            
        if drop_no_dist:
            print('Keeping only galaxies with known value of luminosity distance...')
            df=df[df.dist.notna()==True]
            print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
        
        if drop_HyperLeda2:
            print("Dropping galaxies with HyperLeda name=null and flag2=2...")
            df=df.drop(df[(df['HyperLEDA_name'].isna()) & (df['flag2']==2)].index)
            print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.00%}".format(df.shape[0]/or_dim)+' of total' )
        
        
        # ------ Add useful columns
        
        
        if get_cosmo_z:
            print('Computing cosmological redshifts from given luminosity distance with H0=%s, Om0=%s...' %(cosmo.H0, cosmo.Om0))

            
            z_max = df[df.dist.notna()]['z'].max() +0.01
            z_min = max(0, df[df.dist.notna()]['z'].min() - 1e-05)
            print('Interpolating between z_min=%s, z_max=%s' %(z_min, z_max))
            z_grid = np.linspace(z_min, z_max, 200000)
            dL_grid = cosmo.luminosity_distance(z_grid).value
            
            if not drop_no_dist:
                dLvals = df[df.dist.notna()]['dist']
                print('%s points have valid entry for dist' %dLvals.shape[0])
                zvals = df[df.dist.isna()]['z']
                print('%s points have null entry for dist, correcting original redshift' %zvals.shape[0])
                z_cosmo_vals = np.where(df.dist.notna(), np.interp( df.dist , dL_grid, z_grid), df.z) 
                #z_cosmo_vals = np.interp( df.dist , dL_grid, z_grid)
            
            else:
                z_cosmo_vals = np.interp( df.dist , dL_grid, z_grid)
            
            df.loc[:,'z_cosmo'] = z_cosmo_vals
                
            
            
            
            if not CMB_correct and not group_correct and pos_z_cosmo:
                print('Keeping only galaxies with positive cosmological redshift...')
                df = df[df.z_cosmo >= 0]
                print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
        
        
        if group_correct:
            df_groups =  pd.read_csv(group_cat_path)#, sep=" ", header=None)
            df = group_correction(df, df_groups, which_z_correct)
            
      
        
        if CMB_correct:

            df = CMB_correction(df, which_z_correct)
            if pos_z_cosmo:
                print('Keeping only galaxies with positive cosmological redshift in the CMB frame...')
                df = df[df[which_z_correct+'_corr' ]>= 0]
                print('Kept %s points'%df.shape[0]+ ' or ' +"{0:.0%}".format(df.shape[0]/or_dim)+' of total' )
        
        if err_vals is not None:
            print('Adding errors on z with %s values' %err_vals)
            if err_vals=='GLADE':
                scales = np.where(df['flag2'].values==3, 1.5*1e-04, 1.5*1e-02)
            elif err_vals=='const_perc':
                scales=np.where(df['flag2'].values==3, df[which_z]/100, df[which_z]/10)
            elif err_vals=='const':
                scales = np.full(df.shape[0], 200/clight) 
            else:
                raise ValueError('Enter valid choice for err_vals. Valid options are: GLADE, const, const_perc . Got %s' %err_vals)
            
            df.loc[:, 'Delta_z'] = scales
        
            df.loc[:, "z_lowerbound"] = df[which_z] - 3*df.Delta_z
            df.z_lowerbound[df.z_lowerbound < 0] = 0
            df.loc[:, "z_lower"] = df[which_z] - df.Delta_z
            df.z_lower[df.z_lower < 0.5*df[which_z]] = 0.5*df[which_z]
            df.loc[:, "z_upper"] = df[which_z] + df.Delta_z
            df.loc[:, "z_upperbound"] = df[which_z] + 3*df.Delta_z
           
        if add_B_lum:
            print('Computing total luminosity in B band...')
            # add  a column for B-band luminosity 
            my_dist=cosmo.luminosity_distance(df[which_z].values).value
            df.loc[:,"B_Abs"]=df.B-5*np.log10(my_dist)-25
            BLum = df.B_Abs.apply(lambda x: TotL(x, MBSun))
            df.loc[:,"B_Lum"] =BLum
            df = df.drop(columns='B_Abs')
            df = df.drop(columns='B')
            #print('Done.')
        if add_K_lum:
            print('Computing total luminosity in K band...')
            # add  acolumn for B-band luminosity 

            #df.B_err = df.B_err.fillna(df.B_err.mean()) 
            my_dist=cosmo.luminosity_distance(df[which_z].values).value
            df.loc[:,"K_Abs"]=df.K-5*np.log10(my_dist)-25
            KLum = df.K_Abs.apply(lambda x: TotL(x, MKSun))
            df.loc[:,"K_Lum"]=KLum
            df = df.drop(columns='K_Abs')
            df = df.drop(columns='K')
        
        
        # ------ 
        print('Done.')
        
        return df



# -------------------------------------

def CMB_correction(df, which_z_correct):
            print('Correcting %s for CMB reference frame...' %which_z_correct)
            theta = 0.5 * np.pi - np.deg2rad(df['dec'].values)
            phi = np.deg2rad(df['RA'].values)
            v_gal = clight*df[which_z_correct].values
            phi_CMB, dec_CMB = gal_to_eq(np.radians(l_CMB), np.radians(b_CMB))
            theta_CMB =  0.5 * np.pi - dec_CMB
                        
            delV = v_CMB*(np.sin(theta)*np.sin(theta_CMB)*np.cos(phi-phi_CMB) +np.cos(theta)*np.cos(theta_CMB))
            
            v_corr = v_gal+delV 
            
            z_corr = v_corr/clight
            df.loc[:,which_z_correct+'_corr'] = z_corr
            return df


def group_correction(df, df_groups, which_z_correct):
            df.loc[:, which_z_correct+'_or'] = df[which_z_correct].values        
            
            print('Correcting %s for group velocities...' %which_z_correct)
   
            
            zs = df.loc[df['PGC'].isin(df_groups['PGC'])][['PGC', which_z_correct]] 
            z_corr_arr = []
            #z_group_arr = []
            for PGC in zs.PGC.values:
                #z_or=zs.loc[zs['PGC']==PGC][which_z_correct].values[0]
                PGC1=df_groups[df_groups['PGC']==PGC]['PGC1'].values[0]
                #print(PGC1)
                
                z_group = df_groups[df_groups['PGC1']== PGC1].HRV.mean()/clight
                
                
                z_corr_arr.append(z_group)
            z_corr_arr=np.array(z_corr_arr)
            
            df.loc[df['PGC'].isin(df_groups['PGC']), which_z_correct] = z_corr_arr                        
            correction_flag_array = np.where(df[which_z_correct+'_or'] != df[which_z_correct], 1, 0)
            df.loc[:, 'group_correction'] = correction_flag_array
            
            return df


def gal_to_eq(l, b):
    '''
    input: galactic coordinates (l, b) in radians
    returns equatorial coordinates (RA, dec) in radians
    
    https://en.wikipedia.org/wiki/Celestial_coordinate_system#Equatorial_↔_galactic
    '''
    
    l_NCP = np.radians(122.93192)
    
    del_NGP = np.radians(27.128336)
    alpha_NGP = np.radians(192.859508)
    
    
    RA = np.arctan((np.cos(b)*np.sin(l_NCP-l))/(np.cos(del_NGP)*np.sin(b)-np.sin(del_NGP)*np.cos(b)*np.cos(l_NCP-l)))+alpha_NGP
    dec = np.arcsin(np.sin(del_NGP)*np.sin(b)+np.cos(del_NGP)*np.cos(b)*np.cos(l_NCP-l))
    
    return RA, dec


def compute_Blum(inside_gal_df, band, H0):
    l_name=band+'_Lum'
    inside_gal_df.loc[:, l_name] = inside_gal_df[l_name]/(H0/70)**2
    return inside_gal_df


def compute_Klum(inside_gal_df, band, H0):
    l_name=band+'_Lum'
    inside_gal_df.loc[:, l_name] = inside_gal_df[l_name]/(H0/70)**2
    return inside_gal_df

def TotL(x, MSun=5.498): 
    return 10**(-0.4* (x+25-MSun))


def minmaxz(cosmo, z_min=None, z_max=None, dL_min=None, dL_max=None, Xi0=1, n=1.91):
        
        """
        Converts a pair of values z_min,z_max in corresponding luminosity diststances dL_min, dL_max
        or vice versa
        also gives comoving distances d_min, d_max
        """
        from astropy.cosmology import z_at_value
        import astropy.units as u

        if dL_min is None and dL_max is None and z_min is None and z_max is None:
            raise ValueError('Either (d_min, d_max) or (z_min, z_max) should be specified') 
        
        if z_min is None and z_max is None:
            
            if dL_min!=0:
                z_min = z(dL_min,cosmo,Xi0,n ) #z_at_value(cosmo.luminosity_distance, dL_min * u.Mpc)
            else: 
                z_min=0
            z_max = z(dL_max,cosmo,Xi0,n ) #z_at_value(cosmo.luminosity_distance, dL_max * u.Mpc)
        
        elif dL_min is None and dL_max is None:
            if z_min!=0:
                dL_min = dL_GW(cosmo, z_min, Xi0, n)#cosmo.luminosity_distance(z_min).value
            else: 
                dL_min=0
            dL_max = dL_GW(cosmo, z_max, Xi0, n)#cosmo.luminosity_distance(z_max).value
        
        else:
            raise ValueError('Specify (dL_min, dL_max) or (z_min, z_max) ')
        
        return z_min, z_max,  dL_min, dL_max


def com_vol(cosmo,
            theta_min=0, theta_max=np.pi, phi_min=0, phi_max=2*np.pi,
            z_min=None, z_max=None):
        """
        Returns comoving volume in Mpc^3
        """
        import scipy.integrate as integrate   
        if z_min is None and z_max is None:
            raise ValueError(' (z_min, z_max) should be specified') 
                
        phi_fac = phi_max-phi_min
        theta_fac = integrate.quad(lambda x: np.sin(x), theta_min, theta_max)[0]
        vol_fac = integrate.quad(lambda x: cosmo.differential_comoving_volume(x).value , z_min, z_max)[0]
    
        return vol_fac*theta_fac*phi_fac

    
def dL_GW(cosmo, z, Xi0=1, n=1.91):
    '''
    Modified GW luminosity distance
    '''
    return (cosmo.luminosity_distance(z).value)*Xi(z, Xi0, n) 


def dL_GW_from_dL(cosmo, ref_cosmo, dL,  Xi0=1, n=1.91):  
    '''
    Modified GW luminosity distance from luminosity distance given reference cosmology
    '''
    z_val = z(dL, ref_cosmo, Xi0=1, n=1.91, H0=None)
    
    return dL_GW(cosmo, z_val, Xi0=Xi0, n=n)
 
      
def Xi(z, Xi0=1, n=1.91):

    return Xi0+(1-Xi0)/(1+z)**n


def z(dL_GW_val, cosmo, Xi0=1, n=1.91, H0=None): 
    '''
    Returns redshift for a given luminosity distance dL_GW_val (in Mpc)
    
    Input:
        - dL_GW_val luminosity dist in Mpc
        - cosmo: astropy.cosmology.FlatLCDM object
        - Xi0: float. Value of Xi_0
        - n: float. Value of n
        - H0: float. Value of H0 . If not None, the solver will give priority to this and
                                    fix Xi0=1
    '''
    if H0 is not None:
        Xi0=1
    from scipy.optimize import fsolve
    #print(cosmo.H0)
    func = lambda z : dL_GW(cosmo, z, Xi0, n) - dL_GW_val
    z = fsolve(func, 0.5)
    return z[0]
 
    
# ---------------------------------------------

def hav(theta):
    return (np.sin(theta/2))**2    
 
def haversine(phi, theta, phi0, theta0):
    return np.arccos(1-2*(hav(theta-theta0)+hav(phi-phi0)*np.sin(theta)*np.sin(theta0)))

def get_thphicone(center, psi):
    '''
    center = (RA, dec) of center
    psi = ang opening in radians
    '''
    phi0, theta0 = center[0]*(np.pi/180), np.pi/2-(center[1]*(np.pi/180))
    
    theta_min, theta_max = np.sort([(np.sign(theta0-psi)*np.abs(theta0-psi))%(np.pi), (np.sign(theta0+psi)*np.abs(theta0+psi))%(np.pi)])
    #theta_max = (np.sign(theta0+psi)*np.abs(theta0+psi))%(np.pi)
    
    phi_min, phi_max = np.sort([(np.sign(phi0-psi)*np.abs(phi0-psi))%(2*np.pi), (np.sign(phi0+psi)*np.abs(phi0+psi))%(2*np.pi)])
    #phi_max = (np.sign(phi0+psi)*np.abs(phi0+psi))%(2*np.pi)

    return theta_min, theta_max, phi_min, phi_max
 


def get_GW_metadata(event, host='https://www.gw-openscience.org', **params):
    from gwosc import api
    print('Fetching metadata for '+event+'...')
    res = api.fetch_event_json(event=event, catalog=None, version=None, host=host)
    return res



class Logger(object):
    
    def __init__(self, fname):
        self.terminal = sys.__stdout__
        self.log = open(fname, "w+")
        self.log.write('--------- LOG FILE ---------\n')
        print('Logger created log file: %s' %fname)
        #self.write('Logger')
       
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

    def close(self):
        self.log.close
        sys.stdout = sys.__stdout__


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        print('Got v with type:')
        print(type(v))
        print('Value: %s' %v)
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ---------------------------------------------


def P_complete_truncated(z, P_complete, my_min=1 ):
    return min(P_complete(z), my_min)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def find_median_estimate(probs, x, up_p=0.95, low_p=0.05, digits=1):
    
    idx, my_med = find_nearest(probs, 1/2)
    my_med = x[idx]
    print('idx median: %s' %idx )
    
    idx_low, my_low = find_nearest(probs, low_p)
    low = my_med-x[idx_low]
    
    idx_high, my_high = find_nearest(probs, up_p)
    high = -my_med+x[idx_high]
    
    return np.round(my_med, digits), np.round(low, digits), np.round(high, digits)

 
def highres(x, y, npoints = 1e06):
    sp = UnivariateSpline(x, y, s=0.)
    x_highres = np.linspace(x[0], x[-1], npoints)
    y_highres = sp(x_highres)
    return x_highres, y_highres


def find_mean_estimate(x, p, digits=1):
    
    m = np.trapz(x*p , x)    
    s = np.sqrt(np.trapz(((x-m)**2)*p , x) )
    
    return np.round(m, digits), np.round(s, digits)