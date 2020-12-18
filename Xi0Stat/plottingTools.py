#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 17:15:27 2020

@author: Michi
"""

from globals import *
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'serif'
plt.rcParams["mathtext.fontset"] = "cm"
import numpy as np
import os

import healpy as ho


def plot_completeness(base_path, allGW, catalogue, lims, mask = None, verbose=True):
    c_path = os.path.join(base_path, 'completeness')
    if not os.path.exists(c_path):
        if verbose:
            print('Creating directory %s' %c_path)
        os.makedirs(c_path)
    #if verbose:
    print('Plotting completeness...')
    
    if mask is not None:
        if verbose:
            print("Plotting mask mollview...")
        plt.figure(figsize=(20,10))
        hp.mollview(mask)
        plt.savefig(os.path.join(c_path,'mask.pdf'))
        plt.close()
    plt.close('all')
       
    zslices = np.linspace(0,1,20)
    th, ph = hp.pix2ang(128, np.arange(hp.nside2npix(128)))

    for z in zslices:
        c = catalogue.completeness(th, ph, z)
        plt.figure(figsize=(20,10))
        hp.mollview(c)
        plt.savefig(os.path.join(c_path,'complz={:05.2f}.pdf'.format(z)))
        #plt.clf()
        plt.close()
    plt.close('all')
    
    for key, ev in allGW.items():
        plt.figure(figsize=(20,10))
        #print(ev.find_r_loc())
        zmin, zmax = 0, 1
        mu, l, _,_ = ev.find_r_loc(std_number=2, verbose=False)
    
        z = np.linspace(zmin, zmax, 1000)
    
        c = catalogue.completeness(*ev.find_event_coords(polarCoords=True), z) 
        plt.plot(z, c.T, linewidth=4)
        #plt.show()
        np.savetxt( os.path.join(c_path, key+'_compl_central.txt'), np.array([z, np.squeeze(c)]))
   
        nSamples = 80
        theta, phi, _ = ev.sample_posterior(nSamples=nSamples)
        c = catalogue.completeness(theta, phi, z)
        plt.plot(z, c.T, c='k', alpha=0.1)
        np.savetxt( os.path.join(c_path, key+'_compl_random.txt'), c)
    
        z = np.linspace(ev.zmin, ev.zmax, 100)
        c = catalogue.completeness(*ev.find_event_coords(polarCoords=True), z)
        plt.plot(z, c.T, linewidth=4, c='r')
        plt.title('Completness for ' + key, fontsize=20)
        #plt.legend(['on central event location and in limits of prior range'] + nSamples*['on randomly sampled event locations'] + ['event size for default fiducial cosmology'])
        plt.ylim([0,1.2])
        plt.xlim([zmin, zmax])
        plt.xlabel('z', fontsize=20)
        plt.ylabel(r'$P_{complete}(z)$', fontsize=20)
        plt.savefig(os.path.join(c_path, key+'_completeness.pdf'))
        plt.close()
    plt.close('all')
    #if verbose:
    print('Done.')
    #plt.show()
    
    

    
def plot_post(base_path, grid, post, post_cat, post_compl, event_list,
              band=None,Lcut=None,zR=None,
              myMin=20, myMax=140, 
              yMin=0, yMax=0.025,
              varname='H0', add_value=True):
    
    # post[event]:  posterior, already normalized, already divided by beta
    # post_cat
    # post_compl
    # test_cat_compl: True to plot separately catalogue and hom contributions
    if len(event_list)>1:
        test_cat_compl = False
    elif post_cat is not None and post_compl is not None: 
        test_cat_compl = True
    else:
        test_cat_compl =False
        
    if (len(event_list)>1):
        total_post=np.ones(len(grid)) #initialize an array of dimension equal to that of Xi0_grid, 
                                         #with all entries equal to 1
        total_post_cat=np.ones(len(grid))
        total_post_compl=np.ones(len(grid))

        for event in event_list:
            total_post=total_post*post[event]
            if test_cat_compl:
                total_post_cat=total_post_cat*post_cat[event]
                total_post_compl=total_post_compl*post_compl[event]

        norma=np.trapz(total_post, grid)  
        post['total']=total_post/norma    # add the total postertior as the last entry of the dictionary
        if test_cat_compl:  
            norma_cat=np.trapz(total_post_cat,grid)  
            norma_compl=np.trapz(total_post_compl,grid)
            post_cat['total']=total_post_cat/norma_cat
            post_compl['total']=total_post_compl/norma_compl
    
    fig, ax = plt.subplots(1, figsize=(12,6))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.yaxis.get_offset_text().set_fontsize(14)
    
    #colormap = plt.cm.gist_ncar
    #colors = [plt.cm.Set1(i) for i in np.linspace(0, 1,len(event_list))]
    
    
    for i, event in enumerate(event_list):
        post_grid=post[event]
        ax.plot(grid, post_grid, label = '{}'.format(event)) #color=colors[i]) 
        if test_cat_compl:
            post_grid=post_cat[event]
            ax.plot(grid,post_grid, linestyle='--', dashes=(5,5), label = '{}, cat'.format(event), alpha=0.5, linewidth=0.9) 
            post_grid=post_compl[event]
            ax.plot(grid,post_grid, linestyle='--', dashes=(2,5), label = '{}, compl'.format(event),   alpha=0.5, linewidth=0.9) 

    if (len(event_list)>1):
        ax.plot(grid,post['total'],'k', label = '{}'.format('total'), linewidth=3 )  
    
    ax.plot(grid, np.repeat(1/(grid.max()-grid.min()),grid.shape[0] ), linestyle='dashdot',color='black', alpha=0.3,label = 'prior' )   
    ax.grid(linestyle='dotted', linewidth='0.6')
    ax.set_xlim(myMin, myMax)
    #ax.set_ylim(yMin, yMax)
    
    if varname=='Xi0':
        ax.set_xlabel(r'$\Xi_0$', fontsize=20);
        ax.set_ylabel(r'$p(\Xi_0)$', fontsize=20);
    else:
        ax.set_xlabel(r'$H_0$', fontsize=20);
        ax.set_ylabel(r'$p(H_0)$', fontsize=20);
    ax.legend(fontsize=10);
    #if zR is not None:
    #    ax.set_title('{} band, $L/L_* > $ {}, $z_R =$ {}'.format(band,Lcut,zR), fontsize=20)
    #else:
    if band is not None:
        ax.set_title('{} band, $L/L_* > $ {}'.format(band,Lcut), fontsize=20)
    if add_value:
        if (len(event_list)>1):
            fin_post=post['total']
        else:
            #print(event_list[0])
            fin_post=post[event_list[0]]
        tstr=find_median(fin_post, grid, myMin,myMax, cl=0.9, digits=0)
        fig.suptitle(tstr)
    
    plt.savefig(os.path.join(base_path, 'posterior.pdf'))
    
    
    #plt.show()


    
    return post, post_cat, post_compl




def find_nearest_idx(array, value):
    #for us the array will be the cumulative posterior
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


  
    
def find_median(post, grid, myMin,myMax, cl=0.90, digits=1): 
    
    import scipy.integrate as integrate
    
    grid_finer=np.linspace(myMin, myMax, 500)
    
    post_finer = np.interp(grid_finer, grid, post)
    cumul_post= integrate.cumtrapz(post_finer, grid_finer, initial=0) #cumulative integration of the posterior
    #is an array of the same dimension as H0_grid and post['total'] 
    #initial=0 set to zero the value of the first element of the array cumul_post
    #(if not specified, cumul_post has one element less that post['total'] )

    idx_low= find_nearest_idx(cumul_post, (1-cl)/2 )
    idx_med= find_nearest_idx(cumul_post, 1/2)
    idx_up= find_nearest_idx(cumul_post, (1+cl)/2 ) 
    #note that, it we require  cl =90%, there will be a 5% prob that the result is below Hmin
    #and 5% that it is above Hmax, so we must use (1-cl)/2 in idx_low and  (1+cl))/2 in idx_max

        
    low = np.round(grid_finer[idx_low],digits)
    med = np.round(grid_finer[idx_med],digits)
    up  = np.round(grid_finer[idx_up],digits) 
    err_plus=np.round(up-med,digits)
    err_minus=np.round(med-low,digits)

    print('MEDIAN, HIGH, LOW = {}+{}-{} at {}% c.l., min= {}, max= {}'.format(med,err_plus,err_minus, np.round(100*cl, 1),
                                                                    low,up))
    
    return '{}+{}-{} \n( {}% c.l.)'.format(med,err_plus,err_minus,np.round(100*cl, 1)
                                                                    )

