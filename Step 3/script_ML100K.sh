#!/bin/sh

#SBATCH -o /dev/null
#SBATCH -p NV100q
#SBATCH -w node24
#SBATCH -n 1
#
#
#
python3 hyperOpt_train.py -d "ML-100K" -m "UserKNNCF"
python3 hyperOpt_train.py -d "ML-100K" -m "ItemKNNCF"
python3 hyperOpt_train.py -d "ML-100K" -m "RP3beta"
python3 hyperOpt_train.py -d "ML-100K" -m "WMF"
#
#
#