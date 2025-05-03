#!/bin/bash

tasks=("LaMP-1" "LaMP-2" "LaMP-4" "LaMP-5" "LaMP-7")
llm="llama-3-8b-instruct"
num_retrieve=5

for task in ${tasks[@]}; do
    sbatch \
        --time=48:0:0 \
        --gres=gpu:v100l:1 \
        --mem=64G \
        --output=./logs/sbatch/$llm/bandit_pr-$num_retrieve/$task/output_%j.txt \
        --error=./logs/sbatch/$llm/bandit_pr-$num_retrieve/$task/error_%j.txt \
        scripts/sbatch/train.sh \
            --task $task \
            --llm $llm \
            --num_retrieve $num_retrieve
done
