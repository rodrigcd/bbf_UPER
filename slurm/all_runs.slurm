#!/bin/bash
for i in {0..20}
do
    sbatch scripts/baseline_bbf$i.slurm
done