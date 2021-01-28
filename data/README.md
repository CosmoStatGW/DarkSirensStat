Organization of  ```data/``` folder after running ```./install.sh```:

```bash
	data/
		├── GLADE/
			├──GLADE_2.4.txt
			├──hpx_B_zmin0p01_zmax0p25_nside32_npoints25.txt
			├──hpx_K_zmin0p01_zmax0p25_nside32_npoints25.txt
			
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
						
```