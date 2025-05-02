#!/bin/bash

llm=llama-3-8B-instruct
num_retrieve=5

for task in LaMP-1 LaMP-2 LaMP-3 LaMP-4 LaMP-5 LaMP-7; do
    sbatch scripts/slurm/train.sh \
        --task $task \
        --llm $llm \
        --num_retrieve $num_retrieve
    done
done
