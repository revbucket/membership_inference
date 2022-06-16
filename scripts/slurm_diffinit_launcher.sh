#!/bin/bash

for i in {0..16}
do
    sbatch scripts/slurm_diffinit.sh $i
done

