#!/bin/bash

if [ "$(ls -A Xi0Stat)" ]; then 
	echo "Found Xi0Stat installation"
else
	echo "Do git clone https://github.com/Mik3M4n/Xi0Stat.git and put this script into the outer /Xi0Stat folder" 
	#git clone https://github.com/Mik3M4n/Xi0Stat.git
fi

P="data"

DESFILEPATH="$P/DES/y1a1"
GLADEFILEPATH="$P/GLADE/GLADE_2.4.txt"
GWENSPATH="$P/GWENS/"
O3PATH="$P/GW/O3/"
O2PATH="$P/GW/O2/"

mkdir -p $P/GLADE
mkdir -p $GWENSPATH
mkdir -p $P/DES
mkdir -p $P/misc
mkdir -p $O2PATH
mkdir -p $O3PATH
mkdir -p $P/GW/metadata
mkdir -p $P/GW/detectors

if [ "$(ls -A $P/misc)" ]; then 
	echo "Group catalog is installed."
else
	mv Galaxy_Group_Catalogue.csv $P/misc/.
fi

if [ "$(ls -A $P/GW/detectors)" ]; then 
	echo "Found strain sensitivity data"
else
	echo "Downloading strain sensitivity data..."
	O2EARLYNAME="2016-12-13_C01_L1_O2_Sensitivity_strain_asd.txt"
	O2LATENAME="2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt"
	O3NAME="2018-10-20_DELTAL_FE_L1_O3_Sensitivity_strain_asd.txt"
		
	curl -o $P/GW/detectors/$O3NAME https://dcc.ligo.org/public/0157/G1802165/001/$O3NAME
	curl -o $P/GW/detectors/$O2LATENAME https://dcc.ligo.org/public/0156/G1801952/001/$O2LATENAME
	curl -o $P/GW/detectors/$O2EARLYNAME https://dcc.ligo.org/public/0140/G1700086/002/$O2EARLYNAME
	wget https://www.gw-openscience.org/eventapi/csv/GWTC-1-confident/
	mv index.html $P/GW/metadata/GWTC-1-confident.csv
fi

if [ "$(ls -A $O2PATH)" ]; then 
	echo "Found O2 data"
else
	echo "Downloading O2 data..."
	curl -o o2.tar.gz https://dcc.ligo.org/public/0157/P1800381/007/GWTC-1_skymaps.tar.gz
	tar -xvf o2.tar.gz
	mv GWTC-1_skymaps/* $O2PATH.
	rm -r GWTC-1_skymaps
	rm o2.tar.gz
	
fi

if [ "$(ls -A $O3PATH)" ]; then 
	echo "Found O3 data"
else
	echo "Downloading O3 data..."
	curl -o o3.tar https://dcc.ligo.org/public/0169/P2000223/005/all_skymaps.tar
	tar -xvf o3.tar
	mv all_skymaps/* $O3PATH.
	rm -r all_skymaps
	rm o3.tar
fi

if [ -a $GLADEFILEPATH ]; then
	echo "GLADE is installed in $GLADEFILEPATH" 
else
	echo "Downloading GLADE..." 
	curl -o $GLADEFILEPATH http://aquarius.elte.hu/glade/GLADE_2.4.txt
fi 

if [ -a $DESFILEPATH ]; then
	echo "DES is installed in $DESFILEPATH" 
else
	echo "Downloading DES..." 
	curl -o $DESFILEPATH http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/photoz_catalogs/y1a1_gold_d04_wdnf.fit
fi 

for ((BATCH=0; BATCH < 24; ++BATCH)); do 
	FILENAME="ra_$(printf "%03d" $((BATCH*15)))_$(printf "%03d" $((BATCH*15+15))).csv.gz"
	FILEPATH=$GWENSPATH$FILENAME
	if [ -a $FILEPATH ]; then
		if [[ "$BATCH" -eq 0 ]]; then 
			echo "GWENS-batch $BATCH is installed in $FILEPATH. Suppressing output for other batches" 
		fi
	else
		echo "Downloadin batch $BATCH of GWENS..." 
		URL="https://astro.ru.nl/catalogs/sdss_gwgalcat/"
		URL=$URL$FILENAME
		curl -o $FILEPATH $URL
	fi 
done

if [ -e $P/GLADE/posteriorglade.csv ]
then
    echo "Found GLADE with correct galaxy posteriors"
else
    echo "Processing GLADE for r^2 corrected galaxy posteriors"

    cat << 'ENDOF' > Xi0Stat/Xi0Stat/compGLADEpost.py
from completeness import *
from GLADE import GLADE

skipcompl = SkipCompleteness()

glade = GLADE('GLADE', skipcompl, useDirac=False, computePosterior=True, verbose=True, colnames_final = ['theta','phi','z','z_err', 'z_lower', 'z_lowerbound', 'z_upper', 'z_upperbound', 'w', 'K', 'B_Abs'])

glade.data.to_csv('posteriorglade.csv', index=False)

ENDOF
    python Xi0Stat/Xi0Stat/compGLADEpost.py
    mv posteriorglade.csv $P/GLADE/.
    rm Xi0Stat/Xi0Stat/compGLADEpost.py
fi

