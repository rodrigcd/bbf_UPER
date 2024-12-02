#!/bin/bash
#SBATCH --job-name=UPER_cat_r_25
#SBATCH --output=UPER_cat_r_ensemble/bbf_25_out.log
#SBATCH --error=UPER_cat_r_ensemble/bbf_25_error.log
#SBATCH -p gpu_saxe
#SBATCH -N 1
#SBATCH --mem=10G
#SBATCH --time=0-12:00
#SBATCH --gres=gpu:1

source /usr/share/modules/init/bash 
module load cuda/11.8
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/ceph/apps/ubuntu-20/packages/cuda/11.8.0_520.61.05
eval "$(conda shell.bash hook)"
conda activate bbf2
nvidia-smi
python -c "from jax.lib import xla_bridge;print(xla_bridge.get_backend().platform)"
python -m bbf.train_uper --agent=BBFUPER --gin_files=/nfs/nhome/live/rcdavis/bbf_UPER/bbf/configs/BBFUPER.gin --base_dir=/nfs/nhome/live/rcdavis/bbf_UPER/results/proper_td_error/UPER_cat_r/BeamRider_UPER_cat_r_4 --run_number=4 --game_name=BeamRider --priority_variable=UPER_cat_r 
