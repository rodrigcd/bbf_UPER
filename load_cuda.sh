#!/bin/bash
module load cuda/11.2
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/ceph/apps/ubuntu-20/packages/cuda/11.2.0_460.27.04
