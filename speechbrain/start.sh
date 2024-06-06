#!/bin/bash
cd ../speechbrain/recipes/WSJ0Mix/separation 


CUDA_VISIBLE_DEVICES=0,1,2,3  torchrun --nproc_per_node=4 train.py hparams/sepformer.yaml --data_folder ../../../../wsj0/2speakers 

