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


def plot_completeness(base_path, allGW, catalogue, verbose=True):
    c_path = os.path.join(base_path, 'completeness')
    if not os.path.exists(c_path):
        if verbose:
            print('Creating directory %s' %c_path)
        os.makedirs(c_path)
    #if verbose:
    print('Plotting completeness...')
    for key, ev in allGW.items():
        plt.figure(figsize=(20,10))
        #print(ev.find_r_loc())
        zmin, zmax = ev.get_z_lims()
        mu, l, u, sig = ev.find_r_loc(std_number=2, verbose=False)
        zl = z_from_dLGW(l, H0=70, Xi0=1, n=nGlob)
        zu = z_from_dLGW(u, H0=70, Xi0=1, n=nGlob)
    
        z = np.linspace(zmin, zmax, 10000)
    
        c = catalogue.completeness(*ev.find_event_coords(polarCoords=True), z) 
        plt.plot(z, c, linewidth=4)
        #plt.show()
        np.savetxt( os.path.join(c_path, key+'_base.txt'), c)
   
        nSamples = 80
        theta, phi, _ = ev.sample_posterior(nSamples=nSamples)
        c = catalogue.completeness(theta, phi, z)
        plt.plot(z, c.T, c='k', alpha=0.1)
        np.savetxt( os.path.join(c_path, key+'_random.txt'), c)
    
        z = np.linspace(zl, zu, 100)
        c = catalogue.completeness(*ev.find_event_coords(polarCoords=True), z)
        plt.plot(z, c, linewidth=4, c='r')
        plt.title('Completness for ' + key, fontsize=20)
        #plt.legend(['on central event location and in limits of prior range'] + nSamples*['on randomly sampled event locations'] + ['event size for default fiducial cosmology'])
        #plt.ylim([0,1])
        plt.xlim([0,0.25])
        plt.xlabel('z', fontsize=20)
        plt.ylabel(r'$P_{complete}(z)$', fontsize=20)
        plt.savefig(os.path.join(c_path, key+'_completeness.pdf'))
    #if verbose:
    print('Done.')
    #plt.show()
    
    

    
def plot_post(base_path, grid, post, post_cat, post_compl, event_list,
              band,Lcut,zR,
              myMin=20, myMax=140, 
              yMin=0, yMax=0.025,
              varname='H0',):
    
    # post[event]:  posterior, already normalized, already divided by beta
    # post_cat
    # post_compl
    # test_cat_compl: True to plot separately catalogue and hom contributions
    if len(event_list)>1:
        test_cat_compl = False
    else: test_cat_compl = True
        
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
    if zR is not None:
        ax.set_title('{} band, $L/L_* > $ {}, $z_R =$ {}'.format(band,Lcut,zR), fontsize=20)
    else:
        ax.set_title('{} band, $L/L_* > $ {}'.format(band,Lcut), fontsize=20)
    
    plt.savefig(os.path.join(base_path, 'posterior.pdf'))
    
    
    #plt.show()


    
    return post, post_cat, post_compl


    
