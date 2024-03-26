#!/bin/bash

models=("resnet34" "resnet18")
weights=("default" "none")
train_sets=("train_20_synth" "train_40_synth" "train_80_synth" "train_100_synth")

for model in "${models[@]}"; do
    for weight in "${weights[@]}"; do
        for train_set in "${train_sets[@]}"; do
            python src/models/train.py $model /home/michael/Github/llava_test/mixed_train_sets/$train_set/ /home/michael/Github/llava_test/models/imagenet_100 --epochs 20 --weights $weight
        done
    done
done
