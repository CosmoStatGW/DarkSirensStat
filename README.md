# Statistical method for measuring modified GW propagation with dark sirens

Compute posterior distribution for H0, Xi0


**When making use of this package, please cite [this paper]() and the present git repository.**


## Overview and code organisation


This package is based on the following objects:

* GalCat
* GW
* GWgalaxy


## Usage

### H_0
```
usage: posterior.py [-h] [--data_root DATA_ROOT] [--O2_sm O2_SM]
                    [--meta_path META_PATH] [--out_path OUT_PATH]
                    [--GLADE_PATH GLADE_PATH]
                    [--subset_names SUBSET_NAMES [SUBSET_NAMES ...]]
                    [--B_band_select B_BAND_SELECT]
                    [--K_band_select K_BAND_SELECT] [--add_B_lum ADD_B_LUM]
                    [--add_K_lum ADD_K_LUM] [--drop_z_uncorr DROP_Z_UNCORR]
                    [--get_cosmo_z GET_COSMO_Z] [--CMB_correct CMB_CORRECT]
                    [--group_correct GROUP_CORRECT]
                    [--which_z_correct WHICH_Z_CORRECT] [--which_z WHICH_Z]
                    --parameter PARAMETER [--ks KS [KS ...]] [--k_plot K_PLOT]
                    [--level LEVEL] [--std_number STD_NUMBER]
                    [--complete COMPLETE] [--comp_type COMP_TYPE]
                    [--completion COMPLETION] [--beta_scheme BETA_SCHEME]
                    [--band BAND] [--lum_weighting LUM_WEIGHTING]
                    [--z_err Z_ERR] [--err_vals ERR_VALS]
                    [--count_event_name COUNT_EVENT_NAME]
                    [--count_source_name COUNT_SOURCE_NAME] [--z_flag Z_FLAG]
                    [--n_points_convolution N_POINTS_CONVOLUTION]
                    [--drop_HyperLeda2 DROP_HYPERLEDA2]
                    [--drop_no_dist DROP_NO_DIST]
```

Output will be saved in the folder specified by out\_path. 

Example: H0, GW170817, B band with L_B > 0.6 L_B* , beta from catalogue, CMB and group correction, output saved in result/ :

```
python posterior.py --subset_names 'GW170817' --parameter='H0' --beta_scheme='cat'   --level=0.99 --std_number=3  --B_band_select='False' --K_band_select='False' --z_err='True' --err_vals='GLADE'  --lum_weighting='False'  --complete='True' --completion='uniform'  --which_z='z_cosmo_corr' --which_z_correct='z_cosmo' --group_correct='True' --CMB_correct='True'  --out_path='result/' --band='B' --ks 0.6  --drop_HyperLeda2='True' --drop_no_dist='True'
```

### Xi_0

Usage is the same but with --parameter='Xi0'


## Examples

A tutorial jupyter notebook to compute posteriors in H0 and Xi0 is provided in the notebooks. 

The notebook is an explanatory version of what is done in posterior.py


## Improvements

