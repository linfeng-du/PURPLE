#!/bin/bash

tasks=("LaMP-1" "LaMP-2" "LaMP-3" "LaMP-4" "LaMP-5" "LaMP-7" "LongLaMP-2" "LongLaMP-3" "LongLaMP-4")
num_retrieve=5
llm="phi-4-mini-instruct"

for task in ${tasks[@]}; do
    sbatch \
        --time=48:0:0 \
        --gres=gpu:v100l:1 \
        --mem=64G \
        --output=./logs/sbatch/$llm/bandit_pr-$num_retrieve/$task/%j.out \
        --error=./logs/sbatch/$llm/bandit_pr-$num_retrieve/$task/%j.err \
        scripts/sbatch/train.sh \
            experiment=$llm/bandit_pr-$num_retrieve/$task \
            task=$task \
            num_retrieve=$num_retrieve \
            llm=$llm
done
