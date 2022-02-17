# DarkSirensStat
This package implements a hierarchical bayesian framework for constraining the Hubble parameter and modified GW propagation with dark sirens and galaxy catalogues.

The methods and results can be found in the paper [Cosmology with LIGO/Virgo dark sirens: Hubble parameter and modified gravitational wave propagation](https://arxiv.org/abs/2101.12660). 

Developed by [Andreas Finke](<https://github.com/AndreasFinke>)  and [Michele Mancarella](<https://github.com/Mik3M4n>).


## Summary


* [Citation](https://github.com/CosmoStatGW/Xi0Stat#citation)
* [Installation](https://github.com/CosmoStatGW/Xi0Stat#Installation)
* [Overview and code organisation](https://github.com/CosmoStatGW/Xi0Stat#Overview-and-code-organisation)
* [Data](https://github.com/CosmoStatGW/Xi0Stat#Data)
* [Usage](https://github.com/CosmoStatGW/Xi0Stat#Usage)


## Citation
This package is released together with the paper [Cosmology with LIGO/Virgo dark sirens: Hubble parameter and modified gravitational wave propagation](https://arxiv.org/abs/). When making use of it, please cite the paper and the present git repository. Bibtex:

```
@article{Finke:2021aom,
    author = "Finke, Andreas and Foffa, Stefano and Iacovelli, Francesco and Maggiore, Michele and Mancarella, Michele",
    title = "{Cosmology with LIGO/Virgo dark sirens: Hubble parameter and modified gravitational wave propagation}",
    eprint = "2101.12660",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1088/1475-7516/2021/08/026",
    journal = "JCAP",
    volume = "08",
    pages = "026",
    year = "2021"
}
```

## Installation

First, in a terminal run

```
pip install -r requirements.txt
```
to install the required python libraries.
Then, run

```
./install.sh
```
the code will download all the needed data in the data directory (for its structure, go to data/). These include:

* The GLADE galaxy catalogue
*  O2 and O3 skymaps from the LVC official data releases
*  O2 and O3 strain sensitivities
*  Optionally, the DES and GWENS galaxy catalogues. Do not use this option on a laptop, since the space required is very large

## Overview and code organisation

Here we will soon provide a description of the main logic of the code

## Data
Description is provided inside the data folder.


## Usage


The configuration options are read from the the file ```config.py``` . 
We provide a template with explanation in ```config_template.py```. To creat you own configuration file:

```
cp config_template.py config.py
```
Then, open ```config.py``` and set the options. A description is provided within the file. 

The default options are for running inference for H0 on the O3 BBH events, with flat prior between 20 and 140, mask completeness with 9 masks, interpolation between multiplicative and homogeneous completion, B-band luminosity weights, and a completeness threshold of 50%. The selection effects are computed with MC. To run, execute

```
python main.py
```

The result will be saved in a folder ```results/O2BBHs/``` (the name can be changed in the configuration). 



