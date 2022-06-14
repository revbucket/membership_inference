#!/bin/bash

cd /home/matt/grad/membership_inference

python -m scripts.train_shadowmodel \
    --dataset cifar100 \
    --dataseed 1234 \
    --frac 0.5 \
    --model resnet18 \
    --modelseed 999 \
    --gpus 2 \
    --batch 256 \
    --workers 4 \
    --epochs 100\
    /home/matt/datasets/ \
    test_project \
    test_experiment
