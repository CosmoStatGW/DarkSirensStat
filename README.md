# Xi0Stat
This package implements a hierarchical bayesian framework for constraining the Hubble parameter and modified GW propagation with dark sirens and galaxy catalogues.

The methods and results can be found in the paper [Cosmology with LIGO/Virgo dark sirens: Hubble parameter and modified gravitational wave propagation](https://arxiv.org/abs/). 

Developed by [Andreas Finke](<Andreas.Finke@unige.ch>)  and [Michele Mancarella](<Michele.Mancarella@unige.ch>).


## Summary


* [Citation](https://github.com/CosmoStatGW/Xi0Stat#citation)
* [Installation](https://github.com/CosmoStatGW/Xi0Stat#Installation)
* [Overview and code organisation](https://github.com/CosmoStatGW/Xi0Stat#Overview-and-code-organisation)
* [Data](https://github.com/CosmoStatGW/Xi0Stat#Data)
* [Usage](https://github.com/CosmoStatGW/Xi0Stat#Usage)
* [Modifying the code](https://github.com/CosmoStatGW/Xi0Stat#Modifying-the-code)


## Citation
This package is released together with the paper [Cosmology with LIGO/Virgo dark sirens: Hubble parameter and modified gravitational wave propagation](https://arxiv.org/abs/). When making use of it, please cite the paper and the present git repository. Bibtex:

```
bibex
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



## Modifying the code

