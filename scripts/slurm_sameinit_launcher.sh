#!/bin/bash

for i in {3..16}
do
    sbatch scripts/slurm_sameinit.sh $i
done

