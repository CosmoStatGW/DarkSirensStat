#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 12:21:22 2020

@author: Michi
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from astropy.cosmology import z_at_value
#from astropy.cosmology import FlatLambdaCDM
#import astropy.units as u

#import astropy.units as u
#from astropy.cosmology import z_at_value
import healpy as hp
from utils import *


class GalCat(object):
    
    def __init__(self, path=None, catalogue = None, 
                 colnames=None, sep=" ", col_list=None,
                 cosmo='FlatLCDM', H0=70, Om0=0.27,
                 min_z_th=5e-06,
                 B_band_select=False,K_band_select=False,
                 drop_no_dist=False, 
                 drop_z_uncorr = False,
                add_B_lum=False, MBSun=5.498,#+2.5 *np.log10(0.9)
                add_K_lum = False,
                MKSun = 3.27, get_cosmo_z=False, z_flag=None, 
                CMB_correct=True, which_z_correct='z_cosmo',
                group_correct=False, group_cat_path = '/Users/Michi/Dropbox/statistical_method_schutz_data/data/Galaxy_Group_Catalogue.csv', 
                which_z='z_corr', pos_z_cosmo=True,  
                nbar=0.15, # 0.0198 
                Delta_z=0.05 ,comp_type='local', err_vals='GLADE'):
        
        """
        This class assumes that column names exist with name ['RA','dec','dist','z', 'B_Abs']
        """
        self.Om0=Om0
        self.which_z=which_z
        self.comp_type=comp_type
        
        if catalogue is None:
            print('--- Loading data for catalogue ' + path.split('/')[-1]+ ' ...')
            if colnames is not None:
                #print('Adding colnames')
                self.cat =  pd.read_csv(path, sep=sep, header=None)
                self.cat.columns=colnames
            else:
                #print('Colnames not specified')
                self.cat =  pd.read_csv(path, sep=sep).reset_index(drop=True)
            print('Done.')
            #print(self.cat.head(3))

        else:
            self.cat = catalogue
        
        
        if cosmo=='FlatLCDM':
            from astropy.cosmology import FlatLambdaCDM
            self.cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        else: 
            self.cosmo=cosmo
        
        self.P_complete_dict={}   
        self.z_max = self.cat['z'].max()
        
        
        self.cat = edit_catalogue(self.cat,
                                  col_list=col_list, 
                                  B_band_select=B_band_select,
                                  K_band_select=K_band_select,
                                  z_flag=z_flag,
                                  drop_z_uncorr=drop_z_uncorr,
                                  get_cosmo_z=get_cosmo_z, cosmo=self.cosmo,
                                  drop_no_dist=drop_no_dist,
                                  group_correct=group_correct,
                                  which_z_correct = which_z_correct,
                                  group_cat_path = group_cat_path,
                                  CMB_correct=CMB_correct, l_CMB=l_CMB, b_CMB=b_CMB, v_CMB=v_CMB,
                                  add_B_lum=add_B_lum, MBSun=MBSun,
                                  add_K_lum=add_K_lum, MKSun=MKSun,
                                  which_z=self.which_z, pos_z_cosmo=pos_z_cosmo, err_vals=err_vals)     
        
        
        self.nbar=nbar
        self.Delta_z=Delta_z
        
    # -------------------------
    
            
    
    
    def _get_SchParams_B(self, H0=None):
        cosmo = self._get_cosmo( H0)
        h0 = cosmo.H0.value/100
        LBstar = 2.45*(h0/0.7)**(-2)
        phiBstar = 5.5*10**(-3)*(h0/0.7)**(3)
        return LBstar, phiBstar
        
    def _get_SchNorm(self, phistar, Lstar, alpha, L_star_frac):
        from scipy.special import gammaincc
        from scipy.special import gamma
                
        norm= phistar*Lstar*gamma(alpha+2)*gammaincc(alpha+2, L_star_frac)
        return norm
        
    
    def galaxy_counts(self, 
                      dL_min=None, dL_max=None, z_min=None, z_max=None, 
                      center = (180, 0), 
                      Omega=None, 
                      psi=None,
                      use_hp=False, nside=64,
                      selection='z_corr', # Cut in z or distance
                      norm_method='Schechter', # 'Schechter' or other string
                      band=None,
                      H0= None,#0.679, # If specified, LBstar will be computed from LBstar = 2.45*h07**-2
                      LBstar = None, phiBstar = None,alphaB=-1.07,
                      #LKstar = None, phiKstar = None,
                      # LBstar = 2.60, phiBstar = 5.02 * 1e-3, # Parameters of Schechter function in B band in units of 10^10 solar B band
                      LKstar = 7.57, phiKstar = 6.88 * 1e-3, alphaK=-1.07, # Parameters of Schechter function in K band in units of 10^10 solar B band
                      #norm = 0.0198, # normalization to use if normalizing to full integral over all luminosities
                      L_star_frac = 0., # Fraction of Lstar to cut luminosity
                      Verbose=True):
    
        """
        
        Computes B or K-band luminosity or galaxy density in a shell :
        - between *luminosity distances* dL_min, dL_max or redshifts z_min, z_max
        - Inside a cone with center center=(RA, dec) in degrees and solid angle Omega (in degrees^2 )  
            or angle psi (psi: opening of the cone in radians)
        - Normalized to norm (constant value) or with Schechter function
        - For B-luminosity > L_star_frac x L_* where L_*=1.2*(h)**(-2)
        
        """
        
        LBstar, phiBstar = self._get_SchParams_B(H0=H0)
        cosmo = self._get_cosmo( H0=H0)
        
        full_sky=False
        if psi is None and Omega==4*np.pi*(180/np.pi)**2:
            if Verbose:
                print('Full-sky search')
            full_sky=True
        elif psi>np.pi/2:
            raise ValueError('Angular opening of the cone must not exceed pi/2')

        
        shell, lims, psi = self.slice_cone_search(center=center, 
                          Omega=Omega, psi=psi,
                          z_min=z_min, z_max=z_max, dL_min=dL_min, dL_max=dL_max, 
                          selection=selection, Verbose=Verbose, use_hp=use_hp, nside=nside)
  
        if full_sky:
            theta_min, theta_max, phi_min, phi_max = 0, np.pi, 0, 2*np.pi
        else:
            theta_min, theta_max, phi_min, phi_max = 0,  2*psi, 0, 2*psi

        
        
        z_min, z_max, dL_min, dL_max = lims
        if Verbose:
            print('z and d_L limits:')
            print(z_min, z_max, dL_min, dL_max)
        #theta_min, theta_max, phi_min, phi_max = get_thphicone(center, psi)
        
        Vc = com_vol(cosmo,theta_min=theta_min, theta_max=theta_max, 
                     phi_min=phi_min, phi_max=phi_max, 
                     z_min=z_min, z_max=z_max) 
        
        #Vc = 4*np.pi* cosmo.differential_comoving_volume(z_min).value *self.Delta_z
        
        if norm_method=='Schechter' and band is not None:
            if band=='B':
                phistar, Lstar, alpha = phiBstar, LBstar, alphaB 
            elif band=='K':  
                phistar, Lstar, alpha = phiKstar, LKstar, alphaK
            norm = self._get_SchNorm(phistar, Lstar, alpha, L_star_frac)
            if Verbose:
                print('Using Schechter function normalization with parameters in %s band (phistar, Lstar, alpha )=  %s, %s, %s' %(band, phistar, Lstar, alpha))
        else:
            if Verbose:
                print('Using number counts normalization with n_bar = %s' %self.nbar)
            if band is None:
                Lstar=0
            norm=self.nbar
        
        L_th = L_star_frac*Lstar
        if band=='B':
            shell = shell[shell.B_Lum > L_th ]
        elif band=='K':
            shell = shell[shell.K_Lum > L_th ]
        # galaxy density
        Ngal =shell.shape[0]
        ngalperMpc3= Ngal/Vc
        ngalnorm =  ngalperMpc3/self.nbar
        
        if band is not None:
            if band=='B':
                BLum_tot = shell.B_Lum.sum()
            elif band=='K':
                BLum_tot = shell.K_Lum.sum()
            BLumperMpc3 = BLum_tot/Vc
            BLnorm=BLumperMpc3/(norm) #normalize to the expected average value
        
        
        if Verbose:
            #print('theta_min, theta_max, phi_min, phi_max : %s %s %s %s' %(theta_min, theta_max, phi_min, phi_max))
            print('z_min, z_max : %s, %s' %(z_min, z_max))
            print("vol= % 10.3E Mpc^3" % Vc)
            print("Galaxies with L_B > % 10.3E" %L_star_frac+r'$L_*$')
            print("number of gal = %d" % Ngal)
            print("density= % 10.3E gal/Mpc^3" % ngalperMpc3)
            print("normalized density= %s" % ngalnorm)
            if band is not None:
                print("Total"+band+"-Lum/Mpc^3= % 10.3E " %BLumperMpc3  )
                print("normalized "+band+"-Lum= % 10.3E " %BLnorm  )
        
        if band is not None:
            return BLnorm 
        else:
            return ngalnorm
    
    
    def pixelize(self, z_min=0, z_max=10, 
                   normalize_lum=True, normalize_count=True, 
                   density=True,
                   avg_sub = True, 
                   nside=64,
                   value='count',
                   band=None,
                   norm_method='Schechter', # 'Schechter' or other string
                   H0= None,#0.679, # If specified, LBstar will be computed from LBstar = 2.45*h07**-2
                   LBstar = None, phiBstar = None,alphaB=-1.07,
                   #LKstar = None, phiKstar = None,
                   # LBstar = 2.60, phiBstar = 5.02 * 1e-3, # Parameters of Schechter function in B band in units of 10^10 solar B band
                   LKstar = 7.57, phiKstar = 6.88 * 1e-3, alphaK=-1.07, # Parameters of Schechter function in K band in units of 10^10 solar B band
                   norm = 0.0198, # normalization to use if normalizing to full integral over all luminosities
                   L_star_frac = 0., # Fraction of Lstar to cut luminosity
                   Verbose=True, which_z='z_corr'):
        
        """
        Returns healpix map of galaxy number density or B-band luminosity in a redshift slice
        
        """
        #cosmo = self._get_cosmo( H0)
        
        if norm_method=='Schechter' and band=='B':
            
            LBstar, phiBstar = self._get_SchParams_B(H0)
        elif norm_method=='Schechter' and band=='K' and (LKstar is None or phiKstar is None or alphaK is None):
            raise ValueError('Please provide parameters for Schechter function')
        
        import healpy as hp

        npix = hp.nside2npix(nside)
        angsizepixel = 360**2/(np.pi*npix) #angular size of a pixel in deg^2
        
        
        # Take slice in redshift
        print('---- Slice between %s and %s' %(z_min, z_max))
        print('npix=',npix,' size of pixels=',"{:.3f}".format(angsizepixel),'deg^2')
        df = self.cat[(self.cat[which_z]<z_max)&(self.cat[which_z]>z_min)]   
        
        # Take galaxies with luminosity larger than threshold
        if band=='B':
            L_star = LBstar
            L_band = 'B_Lum'
        elif band=='K':
            L_star = LKstar
            L_band = 'K_Lum'
        elif band==None:
            L_star = 0
            
        df = df[df[L_band] > (L_star_frac*L_star)]
        
        thetas =(np.pi/2) -(df.dec*np.pi/180)
        phis=df.RA*np.pi/180
        indices = hp.ang2pix(nside, thetas, phis)# Go from HEALPix coordinates to indices
        hpxmap = np.zeros(npix, dtype=np.float)#initialize the map with zeros and then fill it
        
        if value=='counts':
            pix_val = 1
        else:                    
            pix_val = df[L_band]
              
            if normalize_lum:
                if norm_method=='Schechter':
                    if band=='B':
                        phistar, Lstar, alpha = phiBstar, LBstar, alphaB 
                    elif band=='K':  
                        phistar, Lstar, alpha = phiKstar, LKstar, alphaK
                norm = self._get_SchNorm( phistar, Lstar, alpha, L_star_frac)
                pix_val=pix_val/(norm) #normalize to the expected luminosity
                 
        np.add.at(hpxmap, indices, pix_val)
        
        if normalize_count:
            hpxmap = hpxmap/hpxmap.mean()
        
        dens_string=''
        if density:
            hpxmap = hpxmap/angsizepixel # now hpxmax contains the number of galaxies(or B_lum) per deg^2
            dens_string='density'
        
        if value=='counts':
            print('Average n.count ' +dens_string+ ': %s' %hpxmap.mean())
        else:
            print('Average '+band+'luminosity '+dens_string+': %s' %hpxmap.mean())
        if avg_sub:
            hpxmap = hpxmap/hpxmap.mean()-1
        
        print('N of galaxy in this slice: %s \n' %df.shape[0])
        
        return hpxmap
    
    
    
    
    def cone_search(self, center, psi, Verbose=True, use_hp=False, nside=64):
        
        """
        Center: coordinates ( RA, dec) of center in deg . then: (phi=RA, theta = pi/2-dec) in rad
        Omega: solid angle of the search in degrees^2 
        psi: opening of the cone in radians
        
        Returns all objects within the cone of given solid angle in a dataframe
        
        """
        
        
        phi0, theta0 = center[0]*(np.pi/180), np.pi/2-(center[1]*(np.pi/180))
        
        if not use_hp:
            #cone_df = self.cat[((self.cat["RA"]*(np.pi/180) -phi0)**2 + (np.pi/2-(self.cat["dec"]*(np.pi/180)) - theta0)**2 ) < psi**2]
            cone_df = self.cat[ haversine(self.cat["RA"]*(np.pi/180), np.pi/2-(self.cat["dec"]*(np.pi/180)), phi0, theta0) < psi]         
        else:
            if Verbose:
                print('Cone search with healpix')
            direction= hp.ang2vec(theta0,phi0)  
            ipix = hp.query_disc(nside, vec=direction, radius=psi)
            cone_df= self.cat[hp.ang2pix(nside, (np.pi/2) -(self.cat.dec*np.pi/180), self.cat.RA*np.pi/180).isin(ipix)]    
        
        return cone_df
        
       
    
    def _psi_from_Omega(self, Omega, Verbose=True):
        """
        Omega: solid angle of the search in degrees^2 
        return psi in rad
        """
        Om_sterad = Omega*(np.pi/180)**2
        Omsrov4pi = np.round(Om_sterad/(4*np.pi), 15)
        if Verbose:
            print(r'Solid angle in sr: %s x 4 $\pi$' %(Omsrov4pi))
        Deltheta = np.arccos(1-2*Omsrov4pi)
        return Deltheta/2
    
    
    def _chech_psi_Om(self, psi, Omega, Verbose=True):
        
        if Omega is None and psi is None:
            raise ValueError('Please enter a valid value for either psi or Omega')
        
        if Omega is not None:
            psi_temp = self._psi_from_Omega(Omega, Verbose=Verbose)
            if psi is not None:
                if psi != psi_temp:
                    raise ValueError('Omega and psi are not compatible! With this Omega I found psi=%s but entered value is %s' %(psi_temp, psi))
                else: psi=psi_temp
            else:
                psi = psi_temp
        
        else:
            Omega = 2*np.pi*(1-np.cos(psi))*(180/np.pi)**2 # in deg^2
            
        if Verbose:
            print('Angular opening: %s rad, %s deg' %(psi, np.degrees(psi)))
            print('Corresponding solid angle: %s deg^2, %s sr, %s x 4 $\pi$' %(Omega, Omega*(180/np.pi)**(-2), Omega*(180/np.pi)**(-2)/(4*np.pi)))
                
        return psi, Omega
    
    
    
    def slice_cone_search(self, center=(180,0), 
                            psi=None, Omega=None,
                          z_min=None, z_max=None, dL_min=None, dL_max=None, 
                          selection='z', 
                          use_hp=True, nside=64,
                          Verbose=True, H0=None):
        
        """
        Center: coordinates ( RA, dec) of center in deg . then: (phi=RA, theta = pi/2-dec) in rad
        Omega: solid angle of the search in degrees^2 
        psi: opening of the cone in radians
        
        Returns all objects within the cone of given solid angle in the dataframe
        
        """
        #print('Selecting galaxies with selection '+selection+' between (%s, %s, %s, %s)' %( z_min, z_max,  dL_min, dL_max))
        cosmo = self._get_cosmo( H0)
        
        if Omega is None and psi is None:
            raise ValueError('Please provide either Omega or psi')
        
        if dL_min is None and dL_max is None and z_min is None and z_max is None:
            raise ValueError('Either (d_min, d_max) or (z_min, z_max) should be specified') 
            
        
        psi, Omega = self._chech_psi_Om(psi, Omega, Verbose=Verbose)
        if Verbose:
            print('Calling minmaxz with z_min, z_max, dL_min, dL_max= %s, %s, %s, %s ' %(z_min, z_max, dL_min, dL_max))
        z_min, z_max,  dL_min, dL_max = minmaxz(self.cosmo, z_min, z_max, dL_min, dL_max)
        lims=(z_min, z_max, dL_min, dL_max)
        if  Verbose:
            print('Output of minmax z: z_min, z_max, dL_min, dL_max = %s, %s, %s, %s'%lims)
        
        if Omega<4*np.pi*(180/np.pi)**2:
            cone_res = self.cone_search(center, psi, Verbose=False, use_hp=use_hp, nside=nside)
        else:
            cone_res = self.cat
        
        
        if selection=='dist':
            if Verbose:
                print('Converting %s to luminosity distance with H0=%s' %(self.which_z,  cosmo.H0.value))
                print('Selection with %s between d_L=%s, %s' %(selection, dL_min, dL_max))  
            r = cosmo.luminosity_distance(cone_res[self.which_z].values).value
            cone_res.loc[:, 'r'] = r
            shell = cone_res[(cone_res.r>dL_min)   & (cone_res.r<=dL_max) ]
        else:
            if Verbose:
                print('Selection with %s between z=%s, %s' %(selection, z_min, z_max))  
            shell = cone_res[(cone_res[selection]>z_min)   & (cone_res[selection]<=z_max) ]
            
        return shell, lims, psi
        
    
    
    def pixelize_z_bins(self, z_min=0, z_max=1.5, delta_z=0.01,
                    **params):
        
        z_edges = np.arange(z_min, z_max+delta_z, delta_z)
        z_slices = [{'z_min':z_edges[i], 'z_max':z_edges[i+1], 
                    
                     'hpmap':self.pixelize(z_min=z_edges[i], z_max=z_edges[i+1], 
                                   **params)}
                         for i in range(len(z_edges)-1)]
        
        return z_slices
 
  
    def f(self, P_complete, z_min=0, z_max=7):
        
        cosmo = self.cosmo #_get_cosmo( H0=H0)
        
        #from scipy.integrate import quad
        #f = 4*np.pi*quad(lambda z: (cosmo.differential_comoving_volume(z).value)*P_complete(z), z_min, z_max)[0]/com_vol(cosmo=cosmo, z_min=z_min, z_max=z_max)
        
        z_grid = np.linspace(z_min, z_max, 200)
        Igrid = np.array([cosmo.differential_comoving_volume(z).value*P_complete(z) for z in z_grid ])
        f = 4*np.pi*np.trapz(Igrid, z_grid)/com_vol(cosmo=cosmo, z_min=z_min, z_max=z_max)
        
        return f
    
    
    def P_miss(self, P_complete, z_min=0, z_max=7, min_p_miss=1e-03):
        cosmo = self.cosmo   
        P_miss = lambda z: np.where( cosmo.differential_comoving_volume(z).value*(1-P_complete(z))/com_vol(cosmo=cosmo, z_min=z_min, z_max=z_max)> min_p_miss, cosmo.differential_comoving_volume(z).value*(1-P_complete(z))/com_vol(cosmo=cosmo, z_min=z_min, z_max=z_max), 0)
        return P_miss
    
    def P_complete(self, comp_type=None,
                   z_min=0, z_max=7, 
                   z_min_th=2e-03, z_max_th= 0.5, step=0.004,
                   band=None, L_star_frac=0., 
                  selection='z_cosmo_corr',  Verbose=False,
                  Delta_z=None, **params):
        
        if comp_type is None:
            comp_type=self.comp_type
        
        if Delta_z is None:
            Delta_z=self.Delta_z
        
        pstr='band'+str(band)+str(L_star_frac)+'z_col_'+str(selection)#+'_lw'+str(lum_weighting)
        
        try: 
            P_complete = self.P_complete_dict[pstr]
        except KeyError:
            print('Computing P_complete in '+str(band)+' band for L>%s L_* ' %L_star_frac+'with selection '+selection ) #+' and lum. weighting=%s' %str(lum_weighting))
            
            P_complete = self._get_P_complete(comp_type=comp_type,z_min=z_min, z_max=z_max, 
                                              z_min_th=z_min_th, z_max_th= z_max_th, step=step,
                                              band=band, L_star_frac=L_star_frac, 
                  selection=selection, Verbose=Verbose,
                   Delta_z=Delta_z, **params)
        
        
        return P_complete
    
    
    def _get_P_complete(self, comp_type=None, 
                        z_min=0, z_max=7, 
                        z_min_th=2e-03, z_max_th= 0.5, step=0.004,
                        band=None, L_star_frac=0., 
                  selection='z_cosmo_corr',  Verbose=False,
                   Delta_z=0.05, **params):
            if comp_type is None:
                comp_type=self.comp_type
            if Delta_z is None:
                Delta_z=self.Delta_z
            print('Using Delta_z = %s' %Delta_z)
            import scipy.interpolate as interpolate
            
            z_grid_0 = np.linspace(1e-20, 0.00199, 20)  
            dL_grid = np.linspace(start=10, stop=1000, num=100)
            dL_grid_1 = np.linspace(start=1000, stop=100000, num=100)[1:]

            z_vec = np.vectorize(lambda x: z(x, self.cosmo, Xi0=1, n=1.91, H0=70) )
            z_grid = z_vec(dL_grid)
            z_grid_1 = z_vec(dL_grid_1)
            z1_grid = np.sort(np.concatenate([z_grid_0, z_grid, z_grid_1]))
            print('Redshift range of the interpolation: %s, %s ' %(z1_grid.min(), z1_grid.max()))
            print('Using %s points' %z1_grid.shape[0])
            print('Try run of galaxy_counts:')
            try_val = self.galaxy_counts(z_min=(1-Delta_z)*z1_grid[2], 
                                            z_max=(1+Delta_z)*z1_grid[2], 
                                            psi=None, Omega=4*np.pi*(180/np.pi)**2,
                                            selection=selection, 
                                            band=band, L_star_frac=L_star_frac, Verbose=True, 
                                             **params) 
            if comp_type=='local':
                print('Local completeness')
                grid = np.array([min(1,self.galaxy_counts(z_min=(1-Delta_z)*z1_grid[j], 
                                            z_max=(1+Delta_z)*z1_grid[j], 
                                            psi=None, Omega=4*np.pi*(180/np.pi)**2,
                                            selection=selection, 
                                            band=band, L_star_frac=L_star_frac, Verbose=Verbose, 
                                             **params) )
                        #if ( (1-Delta_z)*z1_grid[j]>z_min_th and (1+Delta_z)*z1_grid[j]<z_max_th ) else 1 if (1-Delta_z)*z1_grid[j]<z_min_th else 0
                        if (1-Delta_z)*z1_grid[j]>z_min_th else 1
                        for j in range(z1_grid.shape[0])])
            else:
                print('Integrated completeness')
                grid = np.array([min(1,self.galaxy_counts(z_min=0, 
                                            z_max=z1_grid[j], 
                                            psi=None, Omega=4*np.pi*(180/np.pi)**2,
                                            selection=selection, 
                                            band=band, L_star_frac=L_star_frac, Verbose=Verbose, 
                                             **params) )
                        #if ( (1-Delta_z)*z1_grid[j]>z_min_th and (1+Delta_z)*z1_grid[j]<z_max_th ) else 1 if (1-Delta_z)*z1_grid[j]<z_min_th else 0
                        if (1-Delta_z)*z1_grid[j]>z_min_th else 1
                        for j in range(z1_grid.shape[0])])
    
            print('Grid computed')
            P_complete = interpolate.UnivariateSpline(z1_grid, grid, s=0)
            

            pstr='band'+str(band)+str(L_star_frac)+'z_col_'+str(selection)#+'_lw'+str(lum_weighting)
            
                
            self.P_complete_dict[pstr] = lambda x: np.where(P_complete(x)>1, 1, np.where( P_complete(x)<0, 0, P_complete(x))  ) #P_complete_truncated(z, P_complete)
            print('Done - saved with key %s' %pstr)
        
            return P_complete 
    #np.array([np.heaviside(z-self.z_bins[i], 1.)*np.heaviside(-z+self.z_bins[i+1], 1.)*f_arr[i] for i in range(self.z_bins.shape[0]-1) ]).sum()
         


    def _get_cosmo(self, H0=None):
        if H0 is None:
            cosmo = self.cosmo
        else:
            from astropy.cosmology import FlatLambdaCDM
            cosmo = FlatLambdaCDM(H0=H0, Om0=self.Om0)
        return cosmo
    


    def _get_dL_GW(self, H0=None, df=None,  Xi0=1, n=1.91, which_z='z_corr'):
        if df is None:
            df=self.cat         
        cosmo  = self._get_cosmo( H0=H0)
         
        dLGWvec = np.vectorize(dL_GW)
        r = dLGWvec(cosmo, df[which_z].values, Xi0=Xi0, n=n)
                
        return r
    
    