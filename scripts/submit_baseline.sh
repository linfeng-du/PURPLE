#!/bin/bash

llm=llama-3-8b-instruct
num_retrieve=5

for task in LaMP-1 LaMP-2 LaMP-3 LaMP-4 LaMP-5 LaMP-7; do
    for retriever in random bm25 contriever; do
            sbatch scripts/slurm/baseline.sh \
                --task $task \
                --llm $llm \
                --retriever $retriever \
                --num_retrieve $num_retrieve
    done
done
