#!/bin/bash


cd /speechbrain/recipes/LibriMix/separation
#python train.py hparams/sepformer-libri2mix.yaml --data_folder=../../../../mnt/Libri2Mix/

python -m torch.distributed.launch --nproc_per_node=4 train.py hparams/sepformer-libri2mix.yaml --data_folder=../../../../mnt/Libri2Mix/ --distributed_launch --distributed_backend='nccl'
