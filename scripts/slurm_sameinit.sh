#!/bin/bash

#SBATCH -J modelinit
#SBATCH -o slurm_logs/modelinit_%j.o
#SBATCH -e slurm_logs/modelinit_%j.e
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 02:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=mjordan@cs.utexas.edu

base=$((16* $1))


for i in {0..16}; do
    modelseed=$(($i + $base))
    python -m scripts.train_shadowmodel \
    --dataset cifar100 \
    --frac 0.5 \
    --model resnet18 \
    --modelseed $modelseed \
    --gpus 4 \
    --batch 1024 \
    --workers 4 \
    --epochs 100\
    /scratch1/05540/mjordan \
    modelinit \
    model$modelseed
done
