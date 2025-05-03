#!/bin/bash

tasks=("LaMP-1" "LaMP-2" "LaMP-3" "LaMP-4" "LaMP-5" "LaMP-7")
retrievers=("random" "bm25" "contriever")
num_retrieve=5

# Local LLMs
for llm in llama-3-8b-instruct phi-4-mini-instruct; do
    for task in ${tasks[@]}; do
        for retriever in ${retrievers[@]}; do
            sbatch \
                --time=3:0:0 \
                --gres=gpu:v100l:1 \
                --mem=64G \
                --output=./logs/sbatch/$llm/$retriever-$num_retrieve/$task/output_%j.txt \
                --error=./logs/sbatch/$llm/$retriever-$num_retrieve/$task/error_%j.txt \
                scripts/sbatch/baseline.sh \
                    --task $task \
                    --llm $llm \
                    --retriever $retriever \
                    --num_retrieve $num_retrieve
        done
    done
done

# API LLMs
llm="gpt-4o-mini"

for task in ${tasks[@]}; do
    for retriever in ${retrievers[@]}; do
        if [ $retriever == "contriever" ]; then
            compute_args="--gres=gpu:v100l:1"
        else
            compute_args="--cpus-per-task=1"
        fi

        sbatch \
            --time=3:0:0 \
            $compute_args \
            --mem=64G \
            --output=./logs/sbatch/$llm/$retriever-$num_retrieve/$task/output_%j.txt \
            --error=./logs/sbatch/$llm/$retriever-$num_retrieve/$task/error_%j.txt \
            scripts/sbatch/baseline.sh \
                --task $task \
                --llm $llm \
                --retriever $retriever \
                --num_retrieve $num_retrieve
    done
done
