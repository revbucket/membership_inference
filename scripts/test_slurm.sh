#!/bin/bash

#SBATCH -J test_slurm
#SBATCH -o test_slurm_%j.o
#SBATCH -e test_slurm_%j.e
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 01:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=mjordan@cs.utexas.edu



python -m scripts.train_shadowmodel \
    --dataset cifar100 \
    --dataseed 1234 \
    --frac 0.5 \
    --model resnet18 \
    --modelseed 999 \
    --gpus 4 \
    --batch 1024 \
    --workers 4 \
    --epochs 200\
    /scratch1/05540/mjordan \
    test_project \
    test_slurm
