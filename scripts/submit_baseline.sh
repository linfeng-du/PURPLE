#!/bin/bash

local_llms=("llama-3-8b-instruct" "phi-4-mini-instruct")
api_llms=("gpt-4o-mini")
tasks=("LaMP-1" "LaMP-2" "LaMP-3" "LaMP-4" "LaMP-5" "LaMP-7")
retrievers=("random" "bm25" "contriever")
num_retrieve=5

for llm in ${local_llms[@]}; do
    for task in ${tasks[@]}; do
        for retriever in ${retrievers[@]}; do
            sbatch \
                --time=3:0:0 \
                --gres=gpu:v100l:1 \
                --mem=64G \
                --output=./logs/sbatch/$llm/$retriever-$num_retrieve/$task/%j.out \
                --error=./logs/sbatch/$llm/$retriever-$num_retrieve/$task/%j.err \
                scripts/sbatch/baseline.sh \
                    experiment=$llm/$retriever-$num_retrieve/$task \
                    task=$task \
                    retriever=$retriever \
                    num_retrieve=$num_retrieve \
                    llm=$llm
        done
    done
done

for llm in ${api_llms[@]}; do
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
                --output=./logs/sbatch/$llm/$retriever-$num_retrieve/$task/%j.out \
                --error=./logs/sbatch/$llm/$retriever-$num_retrieve/$task/%j.err \
                scripts/sbatch/baseline.sh \
                    experiment=$llm/$retriever-$num_retrieve/$task \
                    task=$task \
                    retriever=$retriever \
                    num_retrieve=$num_retrieve \
                    llm=$llm
        done
    done
done
