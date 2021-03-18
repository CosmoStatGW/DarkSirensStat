#!/bin/bash

if [ "$(ls -A DarkSirensStat)" ]; then 
	echo "Found DarkSirensStat installation"
else
	echo "Do git clone https://github.com/CosmoStatGW/DarkSirensStat.git and put this script into the outer /DarkSirensStat folder" 
fi

P="data"

DESFILEPATH="$P/DES/y1a1.fits"
GLADEFILEPATH="$P/GLADE/GLADE_2.4.txt"
GWENSPATH="$P/GWENS/"
O3PATH="$P/GW/O3/"
O2PATH="$P/GW/O2/"

#mkdir -p $P/GLADE
mkdir -p $GWENSPATH
mkdir -p $P/DES
#mkdir -p $P/misc
mkdir -p $O2PATH
mkdir -p $O3PATH
mkdir -p $P/GW/metadata
mkdir -p $P/GW/detectors

echo "Download GLADE (460M)? y/n"
read DLGLADE

echo "Download GWENS (5.1G)? y/n"
read DLGWENS

echo "Download DES 1Y (22G)? y/n"
read DLDES

echo
echo

if [ "$(ls -A $P/misc)" ]; then 
	echo "Group catalog is installed."
else
	echo "Could not find group catalog in $P/misc/Galaxy_Group_Catalogue.csv."
fi

echo

if [ "$(ls -A $P/GW/detectors)" ]; then 
	echo "Found strain sensitivity data"
else
	echo "Downloading strain sensitivity data..."
	#O2EARLYNAME="2016-12-13_C01_L1_O2_Sensitivity_strain_asd.txt"
	#O2LATENAME="2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt"
	
	O2H1NAME="2017-06-10_DCH_C02_H1_O2_Sensitivity_strain_asd.txt"
	02L1NAME="2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt"
	
	O3H1NAME="O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt"
	O3L1NAME="O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt"
		
	curl -o $P/GW/detectors/$O3H1NAME https://dcc.ligo.org/public/0169/P2000251/001/$O3H1NAME
	curl -o $P/GW/detectors/$O3L1NAME https://dcc.ligo.org/public/0169/P2000251/001/$O3L1NAME
	curl -o $P/GW/detectors/$O2H1NAME https://dcc.ligo.org/public/0156/G1801950/001/$O2H1NAME
	curl -o $P/GW/detectors/$02L1NAME https://dcc.ligo.org/public/0156/G1801952/001/$02L1NAME
	#curl -o $P/GW/detectors/$O2EARLYNAME https://dcc.ligo.org/public/0140/G1700086/002/$O2EARLYNAME
	wget https://www.gw-openscience.org/eventapi/csv/GWTC-1-confident/
	mv index.html $P/GW/metadata/GWTC-1-confident.csv
fi

echo

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

echo

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

echo

if [ -a $GLADEFILEPATH ]; then
	echo "GLADE is installed in $GLADEFILEPATH" 
else
    if [ $DLGLADE == "y" ]; then 
        echo "Downloading GLADE..." 
        curl -o $GLADEFILEPATH http://aquarius.elte.hu/glade/GLADE_2.4.txt
    else
        echo "You can install GLADE manually by copying GLADE_2.4.txt to $GLADEFILEPATH. Then, run this script again for a precomputation of galaxy pdfs."
    fi
fi 

echo

if [ -a $DESFILEPATH ]; then
	echo "DES is installed in $DESFILEPATH" 
else
    if [ $DLDES == "y" ]; then 
	    echo "Downloading DES..." 
	    curl -o $DESFILEPATH http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/photoz_catalogs/y1a1_gold_d04_wdnf.fit
    else
        echo "You can install DES manually by copying y1a1_gold_d04_wdnf.fit to $DESFILEPATH"
    fi
	
fi 

foundbatch="no"
missingbatch="yes"

echo

for ((BATCH=0; BATCH < 24; ++BATCH)); do 
	FILENAME="ra_$(printf "%03d" $((BATCH*15)))_$(printf "%03d" $((BATCH*15+15))).csv.gz"
	FILEPATH=$GWENSPATH$FILENAME
	if [ -a $FILEPATH ]; then
		if [ $foundbatch == "no" ]; then 
		    foundbatch="yes"
			echo "GWENS-batch $BATCH is installed in $FILEPATH. Suppressing output for other installed batches." 
		fi
	else
	    if [ $DLGWENS == "y" ]; then
		    echo "Downloading batch $BATCH of GWENS..." 
		    URL="https://astro.ru.nl/catalogs/sdss_gwgalcat/"
		    URL=$URL$FILENAME
		    curl -o $FILEPATH $URL
        else
            if [ $missingbatch == "yes" ]; then
                missingbatch="no"
                echo "You can install GWENS-batch $BATCH in $FILEPATH manually. Supressing output for other batches that are not found."
            fi
        fi
	fi 
done

echo

if [ -e $P/GLADE/posteriorglade.csv ]
then
    echo "Found GLADE with correct galaxy posteriors"
else
    if [ -a $GLADEFILEPATH ]; then
        echo "Processing GLADE for r^2 corrected galaxy posteriors"

        cat << 'ENDOF' > DarkSirensStat/compGLADEpost.py
from completeness import *
from GLADE import GLADE

skipcompl = SkipCompleteness()

glade = GLADE('GLADE', skipcompl, useDirac=False, galPosterior=True, verbose=True, colnames_final = ['theta','phi','z','z_err', 'z_lower', 'z_lowerbound', 'z_upper', 'z_upperbound', 'w', 'K', 'B_Abs', 'dL', 'completenessGoal'])

glade.data.to_csv('posteriorglade.csv', index=False)

ENDOF
        python DarkSirensStat/compGLADEpost.py
        mv posteriorglade.csv $P/GLADE/.
        rm DarkSirensStat/compGLADEpost.py
    else
        echo "Glade not present. Skipping precomputation of galaxy pdfs."
    fi
    
fi

echo
echo "Installation script completed."
