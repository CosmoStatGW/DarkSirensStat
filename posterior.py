#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:44:50 2020

@author: Michi
"""

from GW import *
from galaxy import *
from GWgalaxy import *
from utils import *
import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt


def err_vals_cases(err_vals):
    
    if err_vals=='const':
        return r'$c \sigma_{z} = 200$ km/s'
    elif err_vals=='GLADE':
        return r'$\sigma_{z} = 1.5 \times 1O^{-2}$ photo-z, $1.5 \times 1O^{-4}$ spec-z'
    elif err_vals=='const_perc':
        return r'$\sigma_{z} = 10\% $ photo-z, $1\% $ spec-z'
    


def plot_post(points_grid, GWgal_L_cat, GWgal_L_miss, betas_cat, betas_miss, prior_grid, 
              GWgal, ks, param_name='H0', beta_scheme=None, k_plot=None, count_post=None, 
              count_event_name=None, count_event_source=None, l_band_name=None, z_err=True,
              err_vals='const', lum_weights=False, complete=False, completion='mult'):
    
    n_events=GWgal_L_cat[str(ks[0])].shape[1]
    
    if k_plot is not None:
        ks  = (k_plot,)
    
    import matplotlib.pyplot as plt      
    names=[e for e in GWgal.GWevents.keys()]
    
    fig, (ax, ax_1) = plt.subplots(2,1, figsize=(10,20), sharex=False, sharey=False)
    #fig_1, ax_1 = plt.subplots(1, figsize=(10,8))
    #ax = fig.add_subplot(2,1,1)
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax_1 = fig.add_subplot(2,1,2)
    #ax_1.set_xticks([])
    #ax_1.set_yticks([])
    grids = {str(k): np.empty(GWgal_L_cat[str(ks[0])].shape) for k in ks}
    betas_grids = {str(k): np.empty(GWgal_L_cat[str(ks[0])].shape) for k in ks}
    norms = {str(k): np.empty(GWgal_L_cat[str(ks[0])].shape[1]) for k in ks}
    for k in ks:
        for i in range(n_events):
            grid_miss = ((GWgal_L_miss[str(k)][:,i])/(betas_cat[str(k)][:,i]+betas_miss[str(k)][:,i]))*prior_grid[str(k)][:,i]
            grid_cat = ((GWgal_L_cat[str(k)][:,i])/(betas_cat[str(k)][:,i]+betas_miss[str(k)][:,i]))*prior_grid[str(k)][:,i]
            grid=grid_miss+grid_cat
        
            # Normalize to 1
            norm = np.trapz(grid, points_grid) 
            try:
                grids[str(k)][:,i] = grid
                norms[str(k)] = norm
                betas_grids[str(k)]= betas_cat[str(k)][:,i]+betas_miss[str(k)][:,i]
            except TypeError:
                grids[str(k)] = grid
                norms[str(k)]= norm
                betas_grids[str(k)]= betas_cat[str(k)][:,i]+betas_miss[str(k)][:,i]
            if k_plot is not None:
                lab = names[i]
            else:
                if l_band_name is not None:
                    lab= names[i]+r' $L_{%s}>%s L_{%s *}$' %(l_band_name, k, l_band_name)
                else:
                    lab= names[i]
            ax.plot(points_grid, grid/norm, label= lab, linewidth=1)
            lab += ', $d_{{max}} = {}$ Mpc'.format(GWgal.GWevents[names[i]].d_max())+', $d_{{obs}} = {}$ Mpc'.format(GWgal.GWevents[names[i]].metadata['luminosity_distance'].values[0])+ ', SNR = %s' %GWgal.GWevents[names[i]].metadata['network_matched_filter_snr'].values[0]
            ax_1.plot(points_grid, betas_cat[str(k)][:,i]+betas_miss[str(k)][:,i], label= lab, linewidth=1)
    
    ax.plot(points_grid, prior_grid[str(ks[0])][:,0], label= 'prior', ls='--', color='b', alpha=0.2, linewidth=0.8)
    #ax.plot(H0grid, full_post, label= 'Counterpart', ls='-.')        
    if n_events>1:
        for k in ks:
            total = np.prod(grids[str(k)] , axis=1 )
            norm_total = np.trapz(total, points_grid) 
            ax.plot(points_grid, total/norm_total, label= 'Combined', color='black', alpha=1, linewidth=2)
        
    if count_post is not None:
        norm_count=np.trapz(np.squeeze((count_post[:,0]/count_post[:,1])), points_grid)
        full_count_post = np.squeeze(count_post[:,0]/count_post[:,1]/norm_count)
        ax.plot(points_grid, full_count_post, color='black', ls='-.', alpha=0.4, label= 'Counterpart for %s\nassuming host %s' %(count_event_name, count_event_source))
    #ax.set_ylim(0, 0.03)
    if param_name=='H0':
        x_max_plt = PRIOR_UP_H0
        x_min_plt = PRIOR_LOW_H0
    else:
        x_max_plt = PRIOR_UP
        x_min_plt = PRIOR_LOW
    ax.set_xlim(x_min_plt,x_max_plt)
    ax_1.set_xlim(x_min_plt,x_max_plt)
    ax.set_xlabel(param_name, fontsize=20)
    ax_1.set_xlabel(param_name, fontsize=20)
    ax.set_ylabel(r'$p(%s)$' %param_name, fontsize=20);
    ax_1.set_ylabel(r'$\beta(%s)$' %param_name, fontsize=20);
    ax.legend(fontsize=12);
    ax_1.legend(fontsize=12);
    if l_band_name is not None:
        title=r'$\beta $ scheme: %s ' %beta_scheme + r',  $L_{%s}>%s L_{%s *}$' %(l_band_name, k_plot, l_band_name)
    else:
        title=r'$\beta $ scheme: %s ' %beta_scheme
    title_1 = title
    if z_err:
        title += ', z_error: %s ' %err_vals_cases(err_vals)
    else:
        title += ', no errors on redshift'
    if lum_weights:
        title+= '\n %s-band luminosity weights' %l_band_name
    else:
        title+='\n No luminosity weights'
    if complete:
        title=title+', completion:'+completion
    else:
        title+=', no completion'
    ax.set_title(title, fontsize=16)
    ax_1.set_title(title_1, fontsize=16)
    #ax.set_yscale('log')
    
    return fig, ax, ax_1


def get_grids(log_post_grid):
    
    GWgal_L_cat = log_post_grid[:,0, :]
    GWgal_L_miss = log_post_grid[:,1, :]
    betas_cat = log_post_grid[:,2, :]
    betas_miss = log_post_grid[:,3, :]
    prior_grid = log_post_grid[:,4, :]
    return GWgal_L_cat, GWgal_L_miss, betas_cat, betas_miss, prior_grid


def main():
    
    in_time=time.time()
    
    
    # Input, output, data folders
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default='/Users/Michi/Dropbox/Local/Physics_projects/statistical_method_schutz_data_local/', type=str, required=False)
    parser.add_argument("--O2_sm", default='GWs/O2_skymaps/', type=str, required=False)
    parser.add_argument("--meta_path", default='/Users/Michi/Dropbox/Local/Physics_projects/statistical_method_schutz_data_local/GWs/GWTC-1-confident.csv', type=str, required=False)   
    parser.add_argument("--out_path", default='results/', type=str, required=False)
    parser.add_argument("--GLADE_PATH", default='/Users/Michi/Dropbox/Local/Physics_projects/statistical_method_schutz_data_local/galaxy_catalogues/GLADE_2.4.txt', type=str, required=False)
    
    # GW events
    parser.add_argument('--subset_names', nargs='+', default=None, required=False)

    # GWgal parameters
    parser.add_argument("--B_band_select", default=False, type=str2bool, required=False)
    parser.add_argument("--K_band_select", default=False, type=str2bool, required=False)
    
    parser.add_argument("--add_B_lum", default=True, type=str2bool, required=False)
    parser.add_argument("--add_K_lum", default=True, type=str2bool, required=False)
    
    parser.add_argument("--drop_z_uncorr", default=False, type=str2bool, required=False)
    parser.add_argument("--get_cosmo_z", default=True, type=str2bool, required=False)
    parser.add_argument("--CMB_correct", default=True, type=str2bool, required=False)
    parser.add_argument("--group_correct", default=False, type=str2bool, required=False)

    parser.add_argument("--which_z_correct", default='z_cosmo', type=str, required=False)
    parser.add_argument("--which_z", default='z_cosmo_corr', type=str, required=False)
    
    parser.add_argument("--parameter", default='H0', type=str, required=True)
    
    parser.add_argument('--ks', nargs='+', default=[0.], required=False)
    parser.add_argument("--k_plot", default=None, type=float, required=False)
    
    parser.add_argument("--level", default=0.99, type=float, required=False)
    parser.add_argument("--std_number", default=3., type=float, required=False)
    
    parser.add_argument("--complete", default=False, type=str2bool, required=False)
    parser.add_argument("--comp_type", default='local', type=str, required=False)
    parser.add_argument("--completion", default='uniform', type=str, required=False)
    parser.add_argument("--beta_scheme", default='uniform', type=str, required=False)
    
    parser.add_argument("--band", default=None, type=str, required=False)
    parser.add_argument("--lum_weighting", default=False, type=str2bool, required=False)
    parser.add_argument("--z_err", default=True, type=str2bool, required=False)
    parser.add_argument("--err_vals", default='const', type=str, required=False)
    
    parser.add_argument("--count_event_name", default=None, type=str, required=False)
    parser.add_argument("--count_source_name", default=None, type=str, required=False)
    
    parser.add_argument("--z_flag", default=None, type=int, required=False)
    parser.add_argument("--n_points_convolution", default=200, type=int, required=False)
    
    
    
    
    FLAGS = parser.parse_args()
    FLAGS.ks = [float(k) for k in FLAGS.ks]
    
    if FLAGS.band is None:
        if len(FLAGS.ks)!=1 and FLAGS.ks[0]!=0:
            print('Got ks=%s with band=None. Setting ks=[0.]' %FLAGS.ks)
        FLAGS.ks = [0.]

        
    
    if not FLAGS.z_err:
        FLAGS.err_vals=None
    
    O2_loc = FLAGS.data_root+FLAGS.O2_sm
    
    if not os.path.exists(FLAGS.out_path):
        print('Creating directory %s' %FLAGS.out_path)
        os.makedirs(FLAGS.out_path)
    else:
       print('Using directory %s for output' %FLAGS.out_path)
    
    logfile = FLAGS.out_path+'logfile.txt'
    myLog = Logger(logfile)
    sys.stdout = myLog
    
    
    print('\n -------- Parameters:')
    for key,value in vars(FLAGS).items():
            print (key,value)
    
    
       
    print('\nO2 data located in %s' %O2_loc)
    
    #  ----------- 
    # RUN
    #  -----------

    if FLAGS.subset_names is not None:
        subset=True
    else:
        subset=False
    O2 = get_all_O2(O2_loc, subset=subset, subset_names=FLAGS.subset_names, 
                meta_path=FLAGS.meta_path)
    
    
    
    GWGal_GLADE = GWGalaxy(O2, meta_path=FLAGS.meta_path,
                       galCat_path=FLAGS.GLADE_PATH, colnames=colnames_GLADE, col_list=col_list_GLADE, 
                        H0=H0_GLADE, Om0=Om0_GLADE,
                        B_band_select=FLAGS.B_band_select, K_band_select=FLAGS.K_band_select, 
                       add_B_lum=FLAGS.add_B_lum, add_K_lum=FLAGS.add_K_lum, MBSun=MBsun_val, 
                        drop_z_uncorr=FLAGS.drop_z_uncorr,
                       get_cosmo_z=FLAGS.get_cosmo_z,
                       which_z=FLAGS.which_z, 
                       CMB_correct=FLAGS.CMB_correct, 
                       group_correct=FLAGS.group_correct,
                       which_z_correct=FLAGS.which_z_correct, z_flag=FLAGS.z_flag,
                       npoints_convolution=FLAGS.n_points_convolution, comp_type=FLAGS.comp_type,
                       err_vals=FLAGS.err_vals
                      ) 
    
    if FLAGS.parameter=='H0':
        print('---- INFERENCE FOR H0')
        grid= np.linspace(PRIOR_LOW_H0,PRIOR_UP_H0, 50 )
    elif FLAGS.parameter=='Xi0':
        print('---- INFERENCE FOR Xi_0')
        grid = np.linspace(PRIOR_LOW,PRIOR_UP, 50 )
    else:
        raise ValueError('Parameter name must be H0 or Xi0. Got %s' %FLAGS.parameter)
    
    np.savetxt(FLAGS.out_path+FLAGS.parameter+'_grid.txt', grid)
    
    log_post_grid = {}
        
    for k in FLAGS.ks:
        if FLAGS.band is not None:
            print(' - Using L> %s x L_*' %str(k))
        else:
            print(' - Using all galaxies')
        if FLAGS.parameter=='Xi0':
            log_post_grid[str(k)]  = np.array([GWGal_GLADE.posterior( Xi0=Xi0i, 
                                                             search_method = 'credible_region',  
                                                             level=FLAGS.level, std_number=FLAGS.std_number,
                                                            complete=FLAGS.complete, 
                                                            Verbose=False, beta_scheme=FLAGS.beta_scheme, 
                                                            band=FLAGS.band, L_star_frac=k, position_val='header' ,
                                                            lum_weighting=FLAGS.lum_weighting, z_err=FLAGS.z_err, 
                                                             completion=FLAGS.completion) 
                          for Xi0i in grid])
        else:
            log_post_grid[str(k)]  = np.array([GWGal_GLADE.posterior( H0=H0i, 
                                                             search_method = 'credible_region',  
                                                             level=FLAGS.level, std_number=FLAGS.std_number,
                                                            complete=FLAGS.complete, 
                                                            Verbose=False, beta_scheme=FLAGS.beta_scheme, 
                                                            band=FLAGS.band, L_star_frac=k, position_val='header' ,
                                                            lum_weighting=FLAGS.lum_weighting, z_err=FLAGS.z_err, 
                                                             completion=FLAGS.completion) 
                          for H0i in grid])
    
    if FLAGS.complete: 
        print('Saving plot p_complete...')
        fig, ax = plt.subplots(1, figsize=(7,5))
        zup = np.max(np.array([GWGal_GLADE._get_z_range(event, n=1.91, std_number=FLAGS.std_number, H0=None, Verbose=False, position_val='header')[1] for event in GWGal_GLADE.GWevents.keys() ]))
        z1_grid = np.linspace(start=0.0, stop=zup, num=500)
        for key in GWGal_GLADE.galCat.P_complete_dict.keys():
            z_val =key.split('z_col_')[1]
            P_c = GWGal_GLADE.galCat.P_complete_dict[key]
            f = GWGal_GLADE.galCat.f(P_c, z_min=1e-06, z_max=zup)
            vals_grid = P_c(z1_grid)
            band=key.split('band')[1][0]
            frac=key.split('z')[0].split(band)[1]
            ax.plot(z1_grid, vals_grid,  label=r'$L_{%s}>%s  L_{%s  *}$, f=%s, z_col: ' %(band, frac, band, np.round(f,2))+z_val ) 
            
            fname_p_comp = FLAGS.out_path+'P_complete_'+key+'.txt'
            np.savetxt(fname_p_comp, np.array([z1_grid, vals_grid]).T , delimiter=' ')
        ax.set_xlabel('z', fontsize=20);
        ax.set_ylabel(r'$P_{complete}(z)$', fontsize=20);
        #ax.set_xscale('log');
        ax.set_title('%s completeness' %FLAGS.comp_type)
        ax.set_xlim(0,zup);
        ax.set_ylim(0,1.1);
        ax.legend();
        plt.savefig(FLAGS.out_path+'p_complete.png')
    
       
        print('Saving plot p_miss...')
        fig, ax = plt.subplots(1, figsize=(7,5))
        z1_grid = np.linspace(start=0.0, stop=1, num=2000)
        for key in GWGal_GLADE.galCat.P_complete_dict.keys():
            z_val =key.split('z_col_')[1]
            P_c = GWGal_GLADE.galCat.P_complete_dict[key]
            f = GWGal_GLADE.galCat.f(P_c, z_min=1e-06, z_max=zup)
            P_miss = GWGal_GLADE.galCat.P_miss(P_c, z_min=1e-06, z_max=zup)
            vals_grid = P_miss(z1_grid)/(1-f)
            band=key.split('band')[1][0]
            frac=key.split('z')[0].split(band)[1]
            ax.plot(z1_grid, vals_grid,  label=r'$L_{%s}>%s  L_{%s  *}$, f=%s, z_col: ' %(band, frac, band, np.round(f,2))+z_val ) 
        
        ax.set_xlabel('z', fontsize=20);
        ax.set_ylabel(r'$P_{miss}(z) $', fontsize=20);
        ax.set_yscale('log');
        ax.set_xlim(0,zup);
        ax.legend();
        ax.set_title('%s completeness' %FLAGS.comp_type)
        plt.savefig(FLAGS.out_path+'p_miss.png')
        
        
    
    if FLAGS.count_event_name is not None:
        print('\nAdding counterpart for the event %s' %FLAGS.count_event_name)
        print('The host galaxy is %s' %FLAGS.count_source_name)
        if FLAGS.parameter=='Xi0':
            count_post = np.array([GWGal_GLADE.P_counterpart( FLAGS.count_event_name, FLAGS.count_source_name, H0=None, Xi0=Xi0i, beta_scheme=FLAGS.beta_scheme) for Xi0i in grid])
        else:
            count_post = np.array([GWGal_GLADE.P_counterpart( FLAGS.count_event_name, FLAGS.count_source_name, H0=H0i, Xi0=1, beta_scheme=FLAGS.beta_scheme) for H0i in grid])
        fname_count = FLAGS.out_path+'res_'+FLAGS.count_event_name+'_counterpart.txt'
        print('Saving %s ... ' %fname_count)
        np.savetxt(fname_count, np.squeeze(count_post), delimiter=' ')
    else:
        count_post=None
        
    GWgal_L_cat = {}
    GWgal_L_miss = {}
    betas_cat = {}
    betas_miss = {}
    prior_grid = {}
    

    for k in FLAGS.ks:
        
        for i, event in enumerate(GWGal_GLADE.GWevents.keys()):
            fname = FLAGS.out_path+'res_'+event+'_'+str(k)+'.txt'
            print('Saving %s ... ' %fname)
            np.savetxt(fname, log_post_grid[str(k)][:,:,i], delimiter=' ')
        
        
        GWgal_L_cat[str(k)], GWgal_L_miss[str(k)], betas_cat[str(k)],betas_miss[str(k)], prior_grid[str(k)] = get_grids(log_post_grid[str(k)])
    
    for k in FLAGS.ks:
        fig, ax, ax_1 = plot_post(grid, GWgal_L_cat, GWgal_L_miss, betas_cat, betas_miss, prior_grid,
                        GWGal_GLADE, FLAGS.ks, 
                        param_name=FLAGS.parameter, beta_scheme=FLAGS.beta_scheme, k_plot=k, 
                        count_post=count_post, count_event_name=FLAGS.count_event_name, 
                        count_event_source=FLAGS.count_source_name,
                        l_band_name=FLAGS.band, err_vals=FLAGS.err_vals, 
                        lum_weights=FLAGS.lum_weighting, 
                        complete=FLAGS.complete,
                        completion=FLAGS.completion, z_err=FLAGS.z_err)
        if FLAGS.band is None:
            post_beta_path=FLAGS.out_path+'posterior_beta_k'
            post_path = FLAGS.out_path+'posterior_k'
            beta_path = FLAGS.out_path+'beta_k'
        else:
            post_beta_path=FLAGS.out_path+'posterior_beta_k'+str(k)
            post_path = FLAGS.out_path+'posterior_k'+str(k)
            beta_path = FLAGS.out_path+'beta_k'+str(k)
        
        fig.savefig(post_beta_path+'.png')
        #plt.savefig(FLAGS.out_path+'posterior.png')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #fig.savefig('ax2_figure.png', bbox_inches=extent)

        # Pad the saved area by 10% in the x-direction and 20% in the y-direction
        fig.savefig(post_path+ '.png', bbox_inches=extent.expanded(1.1, 1.2))
    
        extent_1 = ax_1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(beta_path+ '.png', bbox_inches=extent_1.expanded(1.1, 1.2))
    
    
    #  ----------- 
    # END AND CLOSE
    #  -----------
    
    print('\nDone in %.2fs' %(time.time() - in_time))
    
    sys.stdout = sys.__stdout__
    myLog.close()
    
    
    
    
if __name__=='__main__':
    
    main()