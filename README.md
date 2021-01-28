# Xi0Stat
This package implements a hierarchical bayesian framework for constraining the Hubble parameter and modified GW propagation with dark sirens and galaxy catalogues.

The methods and results can be found in the paper [Cosmology with LIGO/Virgo dark sirens: Hubble parameter and modified gravitational wave propagation](https://arxiv.org/abs/). 

Developed by [Andreas Finke](<Andreas.Finke@unige.ch>)  and [Michele Mancarella](<Michele.Mancarella@unige.ch>).


## Summary


* [Citation](url#citation)
* [Overview and code organisation](url#Overview-and-code-organisation)
* [Data](url#Data)
	* [Data folders] (url#Data-folders)
* [Usage](url#Usage)
	* [Output](url#Output)
* [Modifying the code](url#Modifying-the-code)
	* [Adding parameters](url#Adding-parameters)

## Citation
This package is released together with the paper [Cosmology with LIGO/Virgo dark sirens: Hubble parameter and modified gravitational wave propagation](https://arxiv.org/abs/). When making use of it, please cite the paper and the present git repository. Bibtex:

```
bibex
```

## Overview and code organisation


The package provides the following modules:



## Data

### Data folders

Organization of  ```data/``` folder:

```bash
data/
		├── GLADE/
			├──GLADE_2.4.txt
			
		├── GW/
			├──O2
				├──GW170809_skymap.fits
				├──GW170608_skymap.fits
				├── ....
			├──O3
				├──GW190413_052954_PublicationSamples.fits
				├──GW190424_180648_PublicationSamples.fits
				├── ....
			├──<future runs names>
				├──<event name 1 skymap>.fits
				├──<event name 2 skymap>.fits
				├── ....
			├──	detectors
				├──2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt
				├──2018-10-20_DELTAL_FE_L1_O3_Sensitivity_strain_asd.txt
				├── future runs strain sensitivities
			├──metadata
			
		├── misc/
			├──Galaxy_Group_Catalogue.csv
			

		├── DES/
			├── <des catalogue>
		├── GWENS/
			├── <GWENS catalogue>
		.
		.
		├── any other catalogue/
			├── <catalogue>
				
		└── planck.txt		
```

## Usage


### Output

## Modifying the code

### Adding parameters

Coming soon ...