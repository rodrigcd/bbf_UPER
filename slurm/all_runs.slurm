#!/bin/bash
for i in {1..20}
do
    sbatch scripts/baseline_bbf$i.slurm
done
