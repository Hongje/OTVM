#!/bin/bash
GPUS=$1

python train_s1_trimap.py --gpu $GPUS
