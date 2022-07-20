#!/bin/bash
GPUS=$1
GPUS_ARRAY=($(echo $GPUS | tr ',' "\n"))
NUMBER_OF_CUDA_DEVICES=${#GPUS_ARRAY[@]}
if [ $NUMBER_OF_CUDA_DEVICES -gt 1 ]; then
    echo "Training with multiple GPUs: $GPUS"
    PY_CMD="-m torch.distributed.launch --nproc_per_node=$NUMBER_OF_CUDA_DEVICES --master_port $((RANDOM + 66000))"
else
    echo "Training with a single GPU: $GPUS"
    PY_CMD=""
fi

python $PY_CMD train.py --stage 4 --gpu $GPUS
