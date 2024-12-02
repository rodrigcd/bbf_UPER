#!/bin/bash
#for i in {1..20}
#do
#    sbatch scripts/baseline_bbf$i.slurm
#done

for i in {0..20}
do
    sbatch UPER_quant_ensemble/ensemble_bbf$i.slurm
done

for i in {0..20}
do
    sbatch UPER_quant_r_ensemble/ensemble_bbf$i.slurm
done

for i in {0..20}
do
    sbatch UPER_cat_ensemble/ensemble_bbf$i.slurm
done

for i in {0..20}
do
    sbatch UPER_cat_r_ensemble/ensemble_bbf$i.slurm
done

for i in {0..20}
do
    sbatch PER_ensemble/ensemble_bbf$i.slurm
done

#for i in {0..20}
#do
#    sbatch default_ensemble/ensemble_bbf$i.slurm
#done

