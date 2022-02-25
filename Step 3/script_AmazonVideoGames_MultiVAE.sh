#!/bin/sh

#SBATCH -o /dev/null
#SBATCH -p DGXq
#SBATCH -w node21
#SBATCH --gres=gpu:1
#SBATCH -n 1
export CUDA_VISIBLE_DEVICES="5"
#
#
#
python3 MultiVAE_train.py -d "Amazon (Video Games)" -n_epochs 200 -num_hidden 1 -beta 0.1
python3 MultiVAE_train.py -d "Amazon (Video Games)" -n_epochs 200 -num_hidden 1 -beta 0.2
python3 MultiVAE_train.py -d "Amazon (Video Games)" -n_epochs 200 -num_hidden 1 -beta 0.3
python3 MultiVAE_train.py -d "Amazon (Video Games)" -n_epochs 200 -num_hidden 1 -beta 0.5
python3 MultiVAE_train.py -d "Amazon (Video Games)" -n_epochs 200 -num_hidden 1 -beta 1.0
#
#
#