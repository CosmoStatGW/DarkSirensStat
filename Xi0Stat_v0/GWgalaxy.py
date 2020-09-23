#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 12:20:33 2020

@author: Michi
"""


import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import erf


from GW import *
from galaxy import *
from utils import *
from keelin import *
import pandas as pd
import scipy.stats

# --------------------------------------  


class GWGalaxy(object):
    
  
    
    def __init__(self, GWevents=None, GWfolder=None, meta_path='/Users/Michi/Dropbox/Local/Physics_projects/statistical_method_schutz_data_local/GWs/GWTC-1-confident.csv',
                 galCat=None, H0=70, Om0=0.27, galCat_path=None, which_z='z_corr', 
                 z_max_int=0.5, z_min_int=1e-05,
                 npoints_convolution=200, 
                 **params):
        
        '''
        GWlist : a list of skymap_3D objects
        GWfolder: folder where GW skymaps are (to be given alternatively to GWlist )
        galCat : a GalCat object
        '''
        if galCat is None and galCat_path is None:
            raise ValueError('Please provide a GalCat object or a path to a valid catalogue')
        elif galCat_path is not None:
            galCat = GalCat(galCat_path, H0=H0, Om0=Om0, which_z=which_z,
                  **params )
        
        if GWevents is None and GWfolder is None:
            raise ValueError('Please provide GW events ')
        
        self.which_z=which_z
        print('As a redshift, I am using the column %s' %self.which_z)
        self.GWevents = GWevents 
        self.galCat = galCat
        
        print('\n --- GW events: ')
        for event in GWevents.keys():
            print(event)
        
        self.z_max_int=z_max_int
        self.z_min_int=z_min_int
        self.npoints_convolution=npoints_convolution
        self.z_min = {}
        self.z_max = {}
        self.inside_gal_df = {}
        self.credible_pixels_dict = {}
        self.idxs={}
        
         
 
    
    def loc_region(self, event_name='GW170817', 
                        levels=[0.9, 0.5], s=0, show_plot=True, 
                        gal_coordinates=None, Delta=5, fsize=(8,8)):
    
        '''
        
        Finds credible levels of given GW event, wiht probabilities given by levels
        levels: a list . e.g levels=[0.9, 0.5]
        
        gal_coordinates: (df['RA'],df['dec'],df['z']) of points we want to add to the plot
    
        Finds probability regions specified by levels (and plots the contours if show_plot=True)
        If gal_coordinates are specified, adds scatter plot of them 
        with color palette specifying redshift
        
        Returns:
            ras, decs, interps, fig, ax
            
            
        
        '''
        
        skymap = self.GWevents[event_name]
    
        try:
            credible_levels = skymap.credible_levels
            cts = skymap.cts
        except AttributeError:
            from ligo.skymap.postprocess import find_greedy_credible_levels
            print('Finding credible levels...')
            skymap.credible_levels = find_greedy_credible_levels(skymap.p)
            credible_levels = skymap.credible_levels
        
            from ligo.skymap.postprocess.contour import contour as LIGO_ct
            print('Computing contours...')
            skymap.cts = { levels[i] : LIGO_ct(credible_levels,  levels, degrees=True)[i] for i in range(len(levels))}
            print('Done.')
            cts = skymap.cts
        
        ras={}
        decs={}
        interps={}
        print('Interpolating contours...')
        for i, level in enumerate(levels):
            ras[str(level)] = [cts[level][0][j][0] for j in range(len(cts[level][0]))]
            decs[str(level)] = [cts[level][0][j][1] for j in range(len(cts[level][0]))]

            tck, u = interpolate.splprep([ras[str(level)], decs[str(level)]], s=s)
            interps[str(level)] = interpolate.splev(u, tck)
            #print('Done.')
        if show_plot:
            fig, ax = self.draw_plot(interps=interps, levels=levels, 
                            gal_coordinates=gal_coordinates, Delta=Delta, fsize=fsize)
            ax.set_title(skymap.event_name, fontsize=22 )
        
        else:
            fig, ax = None, None
        
        
        return ras, decs, interps, fig, ax



    def draw_plot(self, interps=None, levels=None, gal_coordinates=None, 
              Delta=5, r=None, shape='circle', event_coords = None, alpha=0.1, fsize=(8,8)):
        
        
        """
        2D plot in the RA, dec plane around an event specified by event_coords in RA, dec
        
        """

        fig, ax = plt.subplots(1, figsize=fsize)
        cmap = plt.get_cmap("coolwarm")
        colors=['navy','lightseagreen' ]
        
        if gal_coordinates is not None: # (ras, decs, z's)
            if len(gal_coordinates)==4:
                ax.scatter(gal_coordinates[0], gal_coordinates[1], marker='x', c=gal_coordinates[2], cmap=cmap, s = gal_coordinates[3])
            elif len(gal_coordinates)==3:
                ax.scatter(gal_coordinates[0], gal_coordinates[1], marker='x', c=gal_coordinates[2], cmap=cmap)
            else:
                ax.scatter(gal_coordinates[0], gal_coordinates[1], marker='x')
        if interps is not None:
            for i, level in enumerate(levels):
                ax.plot(interps[str(level)][0],interps[str(level)][1], color=colors[i],label="{0:.0%}".format(level)+' probability' )
        if r is not None and shape=='circle':
            #print('Plotting circular shape')
            radius=np.degrees(r) # Remember that the plot is in degrees!
            circle1 = plt.Circle((event_coords[0], event_coords[1]), radius=radius, color=colors[0], alpha=alpha ,label='r = %s' %r)
            ax.add_artist(circle1)
         
        ax.legend()
    
        if gal_coordinates is not None and len(gal_coordinates)>=3:
            from matplotlib.cm import ScalarMappable
            scales = np.linspace(gal_coordinates[2].min(), gal_coordinates[2].max(), 10)
            norm = plt.Normalize(scales.min(), scales.max())
            sm =  ScalarMappable( norm=norm, cmap=cmap)
        
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.ax.set_title("Redshift")
    
        if interps is not None:
            max_level=max(levels)
            x1 = max(interps[str(max_level)][0])
            x2 = min(interps[str(max_level)][0])
        
            y1 = min(interps[str(max_level)][1])
            y2 = max(interps[str(max_level)][1])
        
        elif gal_coordinates is not None:
            x1, x2 = max(gal_coordinates[0]), min(gal_coordinates[0])
            y1, y2 = min(gal_coordinates[1]), max(gal_coordinates[1])
        
        ax.set_ylim(y1-Delta, y2+Delta)
        ax.set_xlim(x1+Delta, x2-Delta)
        
        ax.set_xlabel('Ra', fontsize=20)
        ax.set_ylabel('Dec', fontsize=20)
        
        return fig, ax

    
    
    def find_gal(self, 
             event_name='GW170817', 
             use_credible_regions=True,
             level=0.9, 
             event_coords = None,
             r=None,
             side = None,
             minmax_z = (0. , 1.), 
             plot=False, get_contours=True, levels_plot = [0.9, 0.5], 
             use_hp=False, nside=64, Verbose=False, fsize=(8,8), s=1, Delta=5,
             band='B', L_star_frac=0.25,
             **params):
        '''
        
        If use_credible_regions==True:
            Finds all the galaxies in catalogue that are in the k% proability region with k=level
            of the GW event in skymap and between min and max z
    
        Else:
             Finds all the galaxies in catalogue that are in specified region 
             of the GW event in skymap and between min and max z. 
             The region can be a cone of radius r (if r is not None) or a square of side side (if side is not None)
        
        'At least one of use_credible_regions, radius, side must be specified'

        '''

        
        plot_res={'ras':None, 'decs':None, 'interps':None, 'fig':None, 'ax':None}
        #credible_pixels = None
        
        if use_credible_regions:
            inside_gal_df , plot_res['ras'], plot_res['decs'], plot_res['interps'], plot_res['fig'], plot_res['ax'] = self._find_gal_sm( event_name, level=level, minmax_z = minmax_z, 
                                                get_contours=get_contours, show_plot=plot, levels_plot = levels_plot, fsize=fsize, s=s,  Delta=Delta, Verbose=Verbose, band=band, L_star_frac=L_star_frac, **params)

            
        elif r is not None:
            inside_gal_df, plot_res['fig'], plot_res['ax'] = self.find_gal_coord( event_name=event_name, event_coords=event_coords, r=r, shape='circle',  minmax_z = minmax_z, 
                                                plot=plot, Verbose=Verbose, use_hp=use_hp, nside=nside, fsize=fsize, Delta=Delta, **params)
        elif side is not None:
            inside_gal_df, plot_res['fig'], plot_res['ax'] = self.find_gal_coord(event_name=event_name, event_coords=event_coords, r=side, shape='square',  minmax_z = minmax_z, 
                                                plot=plot, Verbose=Verbose, fsize=fsize, Delta=Delta,band=band, L_star_frac=L_star_frac, **params)
        else:
            raise ValueError('At least one of skymap, radius, side must be specified')
        
        if plot_res['ax'] is not None:
            try:
                plot_res['ax'].set_title(event_name, fontsize=22 )
            except AttributeError:
                pass
    
        return inside_gal_df, plot_res
    
    
    
    def _find_gal_sm(self, event_name='GW170817', 
                level=0.9, 
                minmax_z = (0. , 1.), 
                get_contours=True,
                show_plot=False, levels_plot = [0.9, 0.5], s=0, fsize=(8,8), 
                Delta=5, Verbose=False,
                band='B', L_star_frac = 0.25, LKstar = 7.57, H0=None):
        '''
        Finds all the galaxies in catalogue that are in the k% proability region with k=level
        of the GW event in skymap and between min and max z
    
        '''
        skymap = self.GWevents[event_name]
        minskypdf = skymap.get_credible_region_idx( level=level)
        #min_d, max_d = 
         
        # Select galaxies inside confidence region skymap.p_om(ra_gal, dec_gal)>minskypdf
        if Verbose:
            print(event_name+' - Finding galaxies inside '+"{0:.00%}".format(level)+' probability region and in the redshift range [%s , %s]' %minmax_z ) #+' , or lum distance range [%s , %s]' %(min_d, max_d))
        inside_gal_df_ang = self.galCat.cat[ skymap.p_om(self.galCat.cat['RA'], self.galCat.cat['dec'] ) >= minskypdf ]
        inside_gal_df = inside_gal_df_ang[(minmax_z[0]<=inside_gal_df_ang[self.which_z]) & (inside_gal_df_ang[self.which_z]<=minmax_z[1])]
        
            
        if band=='B' and L_star_frac!=0:
            LBstar, _ = self.galCat._get_SchParams_B(H0=self.galCat.cosmo.H0.value)
            L_th = L_star_frac*LBstar
            inside_gal_df = inside_gal_df[inside_gal_df.B_Lum > L_th ]
            if Verbose:
                print(event_name+' - Using galaxies with L'+band+' > % 10.3E' %L_star_frac+r'$L_*$, or '+band+'_lum > %s' %L_th)
        elif band=='K' and L_star_frac!=0:
            L_th = L_star_frac* LKstar
            inside_gal_df = inside_gal_df[inside_gal_df.K_Lum > L_th ]
            if Verbose:
                print(event_name+' - Using galaxies with L'+band+' > % 10.3E' %L_star_frac+r'$L_*$, or '+band+'_lum > %s' %L_th)
        else:
            L_th=0
            #if Verbose:
            #    print(event_name+' - No selection with luminosity')
            
        
        
        if Verbose:
            print('%s galaxies found ' %inside_gal_df.shape[0])
    
        if get_contours:
            ras, decs, interps, fig, ax = self.loc_region(event_name, levels=levels_plot, 
                                              s=s, show_plot=show_plot,
                                              gal_coordinates=(inside_gal_df['RA'],inside_gal_df['dec'],inside_gal_df['z']), fsize=fsize, Delta=Delta)
        else: 
            ras, decs, interps, fig, ax = None,None,None,None,None 
        
        return inside_gal_df, ras, decs, interps, fig, ax

    
    



    def find_gal_coord(self, event_name='GW170817',
                       event_coords=None,
                       r=1, Omega=None,
                       shape='circle',  
                       minmax_z = (0. , 1.), 
                       plot=False, Verbose=False,
                       use_hp=False, nside=64, fsize=(8,8), Delta=5,  band='B', L_star_frac = 0.25, LKstar = 7.57, H0=None):
        """
        Finds all the galaxies in catalogue that are in a cone
        r : radius of the search in radians (angular opening of the cone)
        Omega: solid angle of the search in degrees^2 
        
        """
        
        if event_name is not None:
            event_coords = self.GWevents[event_name].find_event_coords()
            
        ra_center, dec_center = event_coords[0], event_coords[1]
        
        if Verbose:
            if event_name is None:
                end_str = 'position (RA, dec)='+str(event_coords)+' deg'
            else:
                end_str = event_name
            print('Finding galaxies inside region of radius/side '+ str(np.round(r, 3))+ 'rad with shape '+shape+' and in the redshift range [%s , %s]' %minmax_z +' around '+end_str)
        
        min_d = self.galCat.cosmo.luminosity_distance(minmax_z[0]).value
        max_d = self.galCat.cosmo.luminosity_distance(minmax_z[1]).value
        if shape=='circle':         
            #inside_gal_df = catalogue.loc[ ((catalogue.apply(lambda x: (x['RA'] -ra_center)**2 + (x['dec'] - dec_center)**2 , axis=1) <= r**2) ) & (minmax_z[0]<=catalogue['z']) & (catalogue['z']<=minmax_z[1])]
            
            inside_gal_df, lims, psi = self.galCat.slice_cone_search(center=event_coords, 
                          Omega=Omega, psi=r,
                          z_min=minmax_z[0], z_max=minmax_z[1], #d_min=d_min, d_max=d_max, 
                          selection=self.which_z, Verbose=Verbose, use_hp=use_hp, nside=nside)
                
            
        elif shape=='square':
            inside_gal_df = self.galCat.cat[ ( self.galCat.cat['RA']>= ra_center-r/2)&(self.galCat.cat['RA']<= ra_center+r/2) &( self.galCat.cat['dec']>=dec_center-r/2)&(self.galCat.cat['dec']<= dec_center+r/2) & (min_d<=self.galCat.cat['dist']) & (self.galCat.cat['dist']<=max_d)]
        if Verbose:
            print('%s galaxies found (shape: %s) ' %(inside_gal_df.shape[0], shape))
    
        fig, ax=None, None
        if plot:
            fig, ax = self.draw_plot(interps=None, levels=None, 
                             gal_coordinates=(inside_gal_df['RA'],inside_gal_df['dec'],inside_gal_df['z']),
                             Delta=Delta, 
                             r=r, shape=shape, event_coords = event_coords, fsize=fsize)
            
    
        return inside_gal_df, fig, ax



    def _GWgal_L_cat(self, event_name, Xi0=1, n=1.91, H0=None, 
                     search_method='credible_region',
                     level=0.99, std_number=3, position_val='header' , 
                     norm_to_mean=True, lum_weighting=True, LKstar = 7.57,
                     r_cone = None, max_z=5, n_radii = 2,
                     complete=False, completion='mult',
                     z_err=True, #err_vals='const',
                     Verbose=False, npoints_convolution=None, band=None, L_star_frac=0. ,                   
                      **params):
        '''
        GW likelihood from catalogue for a given event 
        Returns: Likelihood, completeness fraction f, z_min, z_max
        (If complete=False, f=1)
        '''
        
        if H0==PRIOR_LOW_H0 or Xi0==PRIOR_LOW:
            Verbose=True
                
        if npoints_convolution is None:
            npoints_convolution=self.npoints_convolution
        cosmo = self._get_cosmo( H0=H0)
        
        if band is not None:
            l_name=band+'_Lum'
              
        
        if search_method=='credible_region':
            df_ind = event_name+search_method+position_val+str(level)+str(std_number)
            use_credible_regions=True
            r_cone = None
            (z_min, z_max) = self._get_z_range( event_name, n=n, std_number=std_number, H0=H0, 
                        Verbose=False, position_val=position_val)
        else:
            use_credible_regions=False
            if r_cone is None:
                    # max_dist =  haversine(self.cat["RA"]*(np.pi/180), np.pi/2-(self.cat["dec"]*(np.pi/180)), phi0, theta0)
                    # r_cone = max_dist/2
                    area = self.GWevents[event_name].area_p(pp=0.99)*(np.pi/180)**2
                    r_cone = n_radii*np.sqrt(area)
            (z_min, z_max) = (0, max_z)
            df_ind = event_name+search_method+str(max_z)+str(n_radii)+str(r_cone)
        try:
            inside_gal_df = self.inside_gal_df[df_ind]
            #credible_pixels = self.credible_pixels_dict[df_ind]
        except KeyError:
            print('Using df_ind: %s' %df_ind)
            inside_gal_df, _ = self.find_gal(
                                            event_name=event_name, 
                                            use_credible_regions=use_credible_regions,
                                            level=level,  r=r_cone,
                                            minmax_z =  (z_min, z_max), 
                                            plot=False, get_contours=False, 
                                            Verbose=True, band=None, L_star_frac=0,
                                            )
            RAs = inside_gal_df['RA'].values
            decs = inside_gal_df['dec'].values
            my_pixels = self.GWevents[event_name].find_pix(RAs, decs)
            inside_gal_df.loc[:,'pixel' ] = my_pixels
            
            self.inside_gal_df[df_ind] = inside_gal_df
   
        r = self.galCat._get_dL_GW( H0=H0, df=inside_gal_df,  Xi0=Xi0, n=n, which_z=self.which_z)
        inside_gal_df.loc[:, 'r'] = r
        
        # SELECT WITH LUMINOSITY
        if H0==PRIOR_LOW_H0 or Xi0==PRIOR_LOW: 
            Verbose_selection=True
        else:
            Verbose_selection=False
        inside_gal_df = self._select_by_lum(inside_gal_df, band, L_star_frac,  Verbose=Verbose_selection, event_name=event_name)
        
        
        if not complete:
            if Verbose:
                print('Not completing')
            f = 1
        elif completion=='uniform':
            if Verbose:
                print('Using uniform completion')
            P_complete = self.galCat.P_complete( 
                                                 band=band, L_star_frac=L_star_frac, 
                                                 selection=self.which_z, 
                                                 Verbose=False,
                   )
            f = self.galCat.f(P_complete, z_min=z_min, z_max=z_max)
            if H0==PRIOR_LOW_H0 or Xi0==PRIOR_LOW:
                print('f = %s for z_min, z_max = %s, %s' %(f, z_min, z_max))
        else:
            raise NotImplementedError('For completion, only uniform method is available, got %s ' %completion)
        
        if Verbose: 
            print('- Computing likelihood for catalogue part for %s and search method=%s....' %(event_name, search_method))
            
        
        # CONVOLUTION
        if z_err:
            if Verbose:
                print('Convolving with redshift error')
            Lr = self.convolve_lik( event_name, inside_gal_df, H0=H0, Xi0=Xi0, n=n, which_z=self.which_z,
                                         a=(1-0.682689492137)*0.5, nsigma=3)
        else:
            Lr = self.GWevents[event_name].likelihood(inside_gal_df['r'].values, inside_gal_df['RA'].values, inside_gal_df['dec'].values)


                
        if lum_weighting:
            weights = inside_gal_df[l_name].values
            norm = inside_gal_df[l_name].sum()
        else:
            weights = np.ones(Lr.shape)
            norm = inside_gal_df.shape[0]
        
        LL =  f*(np.dot(Lr,weights))/norm  #*self.GWevents[event_name].pixarea
       
        if H0==PRIOR_UP_H0 or Xi0==PRIOR_UP: 
            print('- Done for %s . \n' %event_name)
        return LL, f, z_min, z_max
    
    
    
    def convolve_lik(self, event_name, gals, H0=None, Xi0=1, n=1.91, which_z='z_cosmo_corr',
                 a=(1-0.682689492137)*0.5, nsigma=3, nsampling=50, nkeeling=20) :
            
            '''
            Convolves GW likelihood with redshift distribution
            '''
            
            cosmo=self._get_cosmo(H0=H0)
            
            pix_gals = gals['pixel'].values
            #print(pix_gals)
            
            maxz = 20 
            z_table = np.linspace(0, maxz, 10000)
            
            dL_GW_vec=np.vectorize(lambda x: dL_GW(cosmo, x, Xi0=Xi0, n=n))
            dLgw_table = dL_GW_vec(z_table) 
            #from scipy import interpolate
            #redshFromdLgw = interpolate.interp1d(dLgw_table, z_table, kind='cubic')
            
            skymap = self.GWevents[event_name]
            mu = skymap.mu[pix_gals]
            sigma = skymap.sigma[pix_gals]
            Norm = skymap.norm[pix_gals]
            
            
            rl = mu-nsigma*sigma #dgw.distmu[j] - 3*dgw.distsigma[j]
            rl = np.where(rl>0., rl, 0.)
            rl = np.nan_to_num(rl)
            ru =  mu+nsigma*sigma
            ru = np.nan_to_num(ru)
            
            # grid sampling well the skymap 
            rgrid = np.linspace(rl, ru, nsampling)
            
            # my_skymap is the radial part, a truncated-at-zero normal bounded between rl to ru, 
            # truncated-at-zero normal = gaussian with increased amplitude if part is in the negative 
            my_skymap = 1/(0.5 + 0.5*erf(mu/(sigma*np.sqrt(2))))*norm(mu, sigma).pdf(rgrid)*Norm*skymap.p[pix_gals]

            #zgrid = redshFromdLgw(rgrid).T
            zgrid = np.interp( rgrid , dLgw_table, z_table).T
            

            res_keelin = bounded_keelin_3_discrete_probabilities(zgrid, a, 
                                                                              gals.z_lower.values, gals[which_z].values, 
                                                                              gals.z_upper.values, gals.z_lowerbound.values, 
                                                                              gals.z_upperbound.values,
                                                                                N=nkeeling)

            Lr = np.sum(my_skymap.T*res_keelin, axis=1)

            
            return Lr 
    
   
    def _get_z_range(self, event_name, n=1.91, std_number=3, H0=None, Verbose=False, position_val='header'):
        
        '''
        Computes range in redshift to search for a given event
        '''
        strname = event_name+position_val+str(std_number)
        try:
            z_min, z_max = self.z_min[strname], self.z_max[strname]
        except KeyError:
            if H0 is None:
                #print('Using Xi_0 prior')
                z_min, z_max = self.GWevents[event_name].z_range( Xi0max=PRIOR_UP, Xi0min=PRIOR_LOW,
                                                         n=n, std_number=std_number, Verbose=Verbose, position_val=position_val)
            else:
                #print('Using H0 prior')
                z_min, z_max = self.GWevents[event_name].z_range( H0max=PRIOR_UP_H0, H0min=PRIOR_LOW_H0,
                                                        n=n, std_number=std_number, Verbose=Verbose, position_val=position_val)
            self.z_min[strname], self.z_max[strname] = max(z_min, 0), z_max
        if Verbose: 
            print('Using z_min, z_max with key %s' %strname)
            print(' z_min, z_max = %s, %s' %( str(max(z_min, 0)), str(z_max ) ) )
        
        return max(z_min, 0), z_max
      
    
    def GWgal_L_cat(self, events=None, **params):
        ''' 
        Input: events: list of GW events. If None, default will be self.GWevents (all the GW events in the object)
        Wraps _GWgal_L_cat on all events
        Output: np array of dimension (n_events x 4) where each row is given by the output of _GWgal_L_cat for one event
        '''
        
        if events is None:
            events = self.GWevents
        LLtot=[]
        for event in events:        
            LLtot.append( self._GWgal_L_cat(event, **params))
        
        return np.array(LLtot)
    
      
    def _get_cosmo(self, H0=None):
        '''
        Given H0, returns astropy.cosmology FlatLambdaCDM object
        If H0=None, 70 is used
        '''
        if H0 is None:
            cosmo = self.galCat.cosmo
        else:
            from astropy.cosmology import FlatLambdaCDM
            cosmo = FlatLambdaCDM(H0=H0, Om0=self.galCat.Om0)
        return cosmo
     
    
    def _GWgal_L_miss(self, event_name, Xi0=1, n=1.91, H0=None, std_number=3, level=0.99, 
                      npoints_convolution=1000, Verbose=False,  position_val='header', 
                      lum_weighting=True,
                      band=None, L_star_frac=0., search_method='credible_region',max_z=5,
                      npoints=300,
                      z_min_int=None, z_max_int=None, Delta = 5e-03,
                      **params):
        '''
        GW LIKELIHOOD FROM THE MISSING PART
        '''
        
        if H0==PRIOR_LOW_H0 or Xi0==PRIOR_LOW:
            Verbose=True
        if z_min_int is None:
            z_min_int = self.z_min_int
        if z_max_int is None:
            z_max_int=self.z_max_int
        
        if Verbose: 
            print('- Computing likelihood for missing part for %s and search method=%s....' %(event_name, search_method))
        
        if npoints_convolution is None:
            npoints_convolution=self.npoints_convolution
        
        cosmo = self._get_cosmo( H0=H0)

        
        idxs_str = event_name+str(level)
        try:
            idxs = self.idxs[idxs_str]
        except KeyError:
            print('Computing indexes for %s credible region of %s' %(str(level), str(event_name) ))
            p_th = self.GWevents[event_name].get_credible_region_idx(level=level)
            idxs = self.GWevents[event_name].all_pixels[self.GWevents[event_name].p>p_th]
            self.idxs[idxs_str] = idxs
            print('Saved with key %s' %idxs_str)
    
        
        P_complete = self.galCat.P_complete( 
                                                 band=band, L_star_frac=L_star_frac, 
                                                 selection=self.which_z, 
                                                 Verbose=False,
        
           )
        
        if search_method=='credible_region':
            (z_min, z_max) = self._get_z_range( event_name, n=n, std_number=std_number, H0=H0, 
                        Verbose=Verbose, position_val=position_val)
        else:
            (z_min, z_max) = (0, max_z)
        P_miss = self.galCat.P_miss(P_complete, z_min=z_min, z_max=z_max)
        
        z_grid = np.log10(np.logspace(z_min_int, z_max_int, npoints))
        
     
        if Verbose: 
            print('Computing p_miss between %s , %s with %s point grid' %(min(z_grid), max(z_grid), z_grid.shape[0]))
            
        integrand_grid = np.array([ (P_miss(z))*(self.GWevents[event_name].likelihood_px(dL_GW(cosmo, z, Xi0=Xi0, n=n), idxs)) for z in z_grid])

        integral = np.trapz(integrand_grid, z_grid, axis=0).sum()
        
        #from scipy.integrate import quad
        #integral = quad( ,z_min_int, z_max_int)[0]
        
        if H0<PRIOR_UP_H0:
            Verbose=False
        if Verbose : 
            print('- Done for %s.\n' %event_name)
        return integral #*(1-f)
    
    
    def GWgal_L_miss(self, events=None,z_mins=None, z_maxs=None, **params):
        ''' 
        Input:  - events: list of GW events. If None, default will be self.GWevents (all the GW events in the object)
                - z_mins, z_maxs : lists of min and max redshifts for each of the events
        
        Wraps _GWgal_L_cat on all events
        
        Output: np array of dimension (n_events ) where each row is given by the output of _GWgal_L_miss for one event
        '''
        
        if events is None:
            events = self.GWevents
        LLtot=[]
        for i,event in enumerate(events):  
            z_min, z_max = z_mins[i], z_maxs[i]
            print('Check GWgal_L_miss: for %s, z_min=%s, z_max=%s' %(event, z_min, z_max))
            LLtot.append( self._GWgal_L_miss(event,z_min_int=z_min, z_max_int=z_max, **params))
        
        return np.array(LLtot)
     
        
    def _select_by_lum(self, cat_tmp, band, L_star_frac, LKstar = 7.57, Verbose=True, event_name=''):
        
        '''
        Input:  - galaxy catalogue cat_tmp
                - band B or K
                - L_star_frac luminosity threshold in units of L_*
        Output: galaxy catalogue with galaxies of luminosity in the given band larger than L_star_frac x L_*
        
        '''
        if band=='B':
            LBstar, _ = self.galCat._get_SchParams_B(H0=self.galCat.cosmo.H0.value)
            L_th = L_star_frac*LBstar
            #cat_tmp.loc[:, 'B_Lum_resc'] = cat_tmp['B_Lum']/(H0/70)**2 #compute_Blum(inside_gal_df, band, H0) 
            cat_tmp = cat_tmp[cat_tmp.B_Lum > L_th ]
            if Verbose:               
                #print('Using galaxies with L'+band+' > % 10.3E' %L_star_frac+r'$L_*$, or '+band+'_lum > %s' %L_th)
                print(event_name+' - %s galaxies found with %s lum >%s L* ' %(cat_tmp.shape[0], band, L_star_frac))
                
        elif band=='K':
            L_th = L_star_frac* LKstar
            #cat_tmp.loc[:, 'K_Lum_resc'] = cat_tmp['K_Lum']/(H0/70)**2#compute_Klum(inside_gal_df, band, H0) 
            cat_tmp = cat_tmp[cat_tmp.K_Lum > L_th ]      
            if Verbose:               
                #print('Using galaxies with L'+band+' > % 10.3E' %L_star_frac+r'$L_*$, or '+band+'_lum > %s' %L_th)
                print(event_name+' - %s galaxies found with %s lum >%s L* ' %(cat_tmp.shape[0], band, L_star_frac))
        else:
            if Verbose:
                print('No cut in luminosity.')
                print(event_name+' - %s galaxies found ' %(cat_tmp.shape[0]))
        
        return cat_tmp
        

    
    
    def _get_all_betas_cat(self, fs, cats=None, Xi0=1, H0=None, events=None,                       
                       scheme='cat', band=None, lum_weighting=False, L_star_frac=0., LKstar = 7.57, z_err=True):
        
        '''
        Computes betas from catalogue for all events given H0 or Xi0
       
        Output: np array of betas
        '''
        
        if events is None:
            events = self.GWevents
        btot=[]
        
        cat_tmp = self.galCat.cat
        cat = self._select_by_lum(cat_tmp, band, L_star_frac, LKstar = LKstar, Verbose=False)
       
        for i, event in enumerate(events):  
            
            btot.append( fs[i]*self.GWevents[event].beta(cat=cat, Xi0=Xi0, H0=H0, 
                                                        band=band, lum_weighting=lum_weighting,
                                                        scheme=scheme, which_z=self.which_z, z_err=z_err))
        
        return np.array(btot)
      
        
    
        
        
    
    def _get_all_betas_miss(self, events=None, Xi0=1, H0=None,
                            P_complete=lambda x: 1, dL=None, 
                            z_mins=None, z_maxs=None):
        '''
        Computes betas from missing part for all events given H0 or Xi0
       
        Output: np array of betas
        '''
        
        if events is None:
            events = self.GWevents
        btot=[]
        for i, event in enumerate(events):  
            z_min, z_max = z_mins[i], z_maxs[i]
            print('Check _get_all_betas_miss: for %s, z_min=%s, z_max=%s' %(event, z_min, z_max))
            P_miss=self.galCat.P_miss( P_complete, z_min=z_min, z_max=z_max)
            btot.append( self.GWevents[event].beta_miss(P_miss,z_max=z_max, Xi0=Xi0, H0=H0, dL=dL))
        
        return np.array(btot)
    
    
    
    
    def _get_prior_flat(self, x, p_up, p_low):
        '''
        Flat prior
        '''
        
        return np.where((x>p_up)  & (x<p_low), 0, 1/(p_up-p_low))
            
    
    def posterior(self, Xi0=1,  H0=None, n=1.91, level=0.99, std_number=3, 
                  norm_to_mean=True, events=None, complete=False, completion='uniform',
                  search_method='credible_region',
                  Verbose=False,
                  beta_scheme='cat', band=None, L_star_frac=0.,lum_weighting=False, 
                  z_err=True,
                  **params):
        
        '''
        Posterior for given H0 or Xi0 
        
        Output: GW likelihood from catalogue, GW likelihood from completion, beta from catalogue, beta from completion, prior
        '''
        
        if H0 is None: 
            if Verbose:
                print('############ POSTERIOR FOR XI0 ###############')
            prior = self._get_prior_flat(Xi0, PRIOR_UP, PRIOR_LOW)
        else:
            if Verbose:
                print('############ POSTERIOR FOR H0 ###############')
            prior = self._get_prior_flat(H0, PRIOR_UP_H0, PRIOR_LOW_H0)
        
        
        # 1 CONTRIBUTION TO LIKELIHOOD AND BETA FROM THE CATALOGUE        
        res = self.GWgal_L_cat(events=events, Xi0=Xi0,  n=n, H0=H0, 
                                       level=level, std_number=std_number, 
                                       norm_to_mean=norm_to_mean, Verbose=Verbose, 
                                       complete=complete, completion=completion,
                                       band=band, L_star_frac=L_star_frac,
                                       lum_weighting=lum_weighting, z_err=z_err, 
                                       **params
                                       )
        GWgal_L_cat, fs, z_mins, z_maxs = res[:,0], res[:,1] , res[:,2] , res[:,3]
                          
        betas_cat = self._get_all_betas_cat( fs=fs, Xi0=Xi0, H0=H0, events=events,
                                    scheme=beta_scheme, band=band, 
                                    lum_weighting=lum_weighting, L_star_frac=L_star_frac, z_err=z_err)
        
         
        # 1 CONTRIBUTION TO LIKELIHOOD AND BETA FROM THE COMPLETION
        if complete:
            
            
            #LIKELIHOOD
            if completion=='uniform':
                print('Computing likelihood from completion ...')
                
                GWgal_L_miss = self.GWgal_L_miss(events=events,z_mins=z_mins, z_maxs=z_maxs,Xi0=Xi0, n=n, H0=H0, 
                                             std_number=std_number, level=level, Verbose=Verbose, 
                                             band=band, L_star_frac=L_star_frac, 
                                             lum_weighting=lum_weighting, **params)
            else:
                raise NotImplementedError('For completion, only uniform method is available, got %s' %completion)
                
            
            
            # BETAS
            if beta_scheme=='uniform':
                print('Got beta_scheme=uniform. No correction to beta from missing part required')
                betas_miss = np.zeros(betas_cat.shape)
            else:
                print('Computing betas from completion...')
                P_complete = self.galCat.P_complete( 
                                                 band=band, L_star_frac=L_star_frac, 
                                                 selection=self.which_z, 
                                                 Verbose=False,
                                                 )
                betas_miss = self._get_all_betas_miss(events=events, Xi0=Xi0,  H0=H0, 
                                                            P_complete=P_complete, z_mins=z_mins, z_maxs=z_maxs)
                #print('beta with catalogue from missing part is still to be implemented ! Wrongly using beta_miss=0')
        
        else:
            if Verbose:
                print('Not completing')
            #P_complete = lambda x: x
            GWgal_L_miss = np.zeros(GWgal_L_cat.shape)
            betas_miss = np.zeros(betas_cat.shape)
        

        
        
        return  GWgal_L_cat, GWgal_L_miss, betas_cat, betas_miss, np.array([prior for i in range(GWgal_L_miss.shape[0])])


      
        
    def P_counterpart(self, event_name, count_GWGC_name, H0=None, Xi0=1, n=1.91, beta_scheme='uniform', z_err=True):
        '''
        Likelihood and beta in the counterpart case
        Input:  - GWGC_name of counterpart
                - event name
        '''
        df = self.galCat.cat[self.galCat.cat['GWGC name']==count_GWGC_name]
        r = self.galCat._get_dL_GW( H0=H0, df=df, Xi0=Xi0, which_z=self.which_z)
        df.loc[:, 'r'] = r
        Lr = self.GWevents[event_name].likelihood(df['r'].values, 
                                                     df['RA'].values, 
                                                     df['dec'].values)


        betas_cat = self._get_all_betas_cat( Xi0=Xi0, H0=H0, events=[event_name,],
                                    scheme=beta_scheme, band=None, lum_weighting=False, L_star_frac=0., z_err=z_err)
    
        return  Lr, betas_cat

    
    
   
    

    
    
# --------------------------------------   
 
#  VISUALIZATION TOOLS
    
# --------------------------------------



    def plot_mollview_contours(self, 
                           events=None, 
                           level=0.99, minmax_z = (0, 1),
                           value='Blum',
                           nside=2**7, levels_plot=[0.9], fsize=(14, 12)):
        '''
        Make healpix map of the galaxy catalogue and draws on top of it the contours of confidence regions of GW events
        
        '''
        ######### MAKE HP MAP OF GALAXIES #########
    
        if events is None:
            events = self.GWevents
    
    
        hpxmap = self.galCat.pixelize( z_min=minmax_z[0], z_max=minmax_z[1],
                   nside=nside,
                   Verbose=False, value=value)
    
    
        ######### COMPUTE C.L. OF GW EVENTS IF NOT GIVEN #########
    
        
        inside_gal_dfs={}
        plot_res_ts = {}
        
        for event in events:
            
            print('\n -- Computing CL for '+event+'...')
            print('Position: (%s , %s )' %(self.GWevents[event].find_event_coords()))
            
            inside_gal_dfs[event], plot_res_ts[event] = self.find_gal(event_name=event, 
                                                         use_credible_regions=True,
                                                         level=level, 
                                                         event_coords = None,
                                                         r=None,
                                                         side = None,
                                                         minmax_z = minmax_z, 
                                                         plot=False, get_contours=True, 
                                                         levels_plot = levels_plot, 
                                                         use_hp=True, nside=nside, Verbose=False, 
                                                         fsize=fsize, s=0)
    
        import healpy as hp
        fig = plt.figure(figsize = fsize)
        #fig, ax = plt.subplots(1, figsize=fsize)
        hp.mollview(2+hpxmap,norm='log',title="$%s<z<%s$, equatorial coord" %minmax_z, rot=[180, 0])
        hp.graticule()
        
        from matplotlib.pyplot import cm
        evenly_spaced_interval = np.linspace(0.3, 1, len(events))
        colors = [cm.Reds(x) for x in evenly_spaced_interval] 
        colors_contours = [cm.GnBu(x) for x in evenly_spaced_interval] 
        from matplotlib.lines import Line2D
        markers= [k for k in Line2D.markers.keys()] [:len(events)]
        # Center (check) at ra=12h, dec=0 (i.e. theta=pi/2-dec =0)
        for i,event in enumerate(events): 
            event_coords = self.GWevents[event].find_event_coords()
            interps = plot_res_ts[event]['interps']
            hp.projplot(np.pi/2-interps[str(max(levels_plot))][1]*np.pi/180, interps[str(max(levels_plot))][0]*np.pi/180,linewidth=1,
                        c=colors_contours[i], label=event);

            hp.projscatter((np.pi/2-event_coords[1]*np.pi/180, event_coords[0]*np.pi/180),
                            color=colors[i], marker=markers[i], linewidths=2, label=event);
        plt.legend(fontsize=20)
                           
        return fig,  inside_gal_dfs[event], plot_res_ts[event]
            
            
    
    
# ------------------------------------------------------------------------         
            



def get_all_O2(O2_loc, subset=False, subset_names=['GW170817', 'GW170818', 'GW170814'], 
               **params):
    '''
    Returns dictionary with all skymaps in the folder O2_loc.
    If subset=True, gives skymaps only for the event specified by subset_names
    
    '''
    from os import listdir
    from os.path import isfile, join
    sm_files = [f for f in listdir(O2_loc) if ((isfile(join(O2_loc, f))) & (f!='.DS_Store'))]    
    ev_names = [fname.split('_')[0]  for fname in sm_files]
    if subset:
        ev_names = [e for e in ev_names if e in subset_names]
        sm_files = [e+'_skymap.fits' for e in ev_names]
    print('--- GW events:')
    print(ev_names)
    print('Reading skymaps....')
    all_O2 = {fname.split('_')[0]: skymap_3D(O2_loc+fname, nest=False, **params) for fname in sm_files}
    return all_O2
            