#!/bin/bash
#for i in {1..20}
#do
#    sbatch scripts/baseline_bbf$i.slurm
#done

for i in {0..20}
do
    sbatch UPER_ensemble_quant/ensemble_bbf$i.slurm
done

for i in {0..20}
do
    sbatch UPER_ensemble_quant_r/ensemble_bbf$i.slurm
done

for i in {0..20}
do
    sbatch UPER_ensemble_cat/ensemble_bbf$i.slurm
done

for i in {0..20}
do
    sbatch UPER_ensemble_cat_r/ensemble_bbf$i.slurm
done

for i in {0..20}
do
    sbatch PER_ensemble/ensemble_bbf$i.slurm
done

#for i in {0..20}
#do
#    sbatch default_ensemble/ensemble_bbf$i.slurm
#done

