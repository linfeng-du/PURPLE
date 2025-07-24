#!/bin/bash

ARGS=$(getopt --options "" --long api,llms:,tasks:,retrievers:,num_retrieve:,gpu_type: --name "$0" -- "$@")
eval set -- "$ARGS"

api=0

while true; do
    case "$1" in
        --api) api=1; shift ;;
        --llms) llms="$2"; shift 2 ;;
        --tasks) tasks="$2"; shift 2 ;;
        --retrievers) retrievers="$2"; shift 2 ;;
        --num_retrieve) num_retrieve="$2"; shift 2 ;;
        --gpu_type) gpu_type="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

IFS=',' read -ra llms <<< "$llms"
IFS=',' read -ra tasks <<< "$tasks"
IFS=',' read -ra retrievers <<< "$retrievers"

for llm in ${llms[@]}; do
    for task in ${tasks[@]}; do
        for retriever in ${retrievers[@]}; do
            if [[ $api -eq 0 || $retriever == "contriever" ]]; then
                compute="gres=gpu:$gpu_type:1"
            else
                compute="cpus-per-task=1"
            fi

            experiment="$llm/$retriever-$num_retrieve/$task"
            sbatch \
                --job-name=$experiment \
                --time=3:0:0 \
                --$compute \
                --mem=64G \
                --output=./logs/$experiment/%j.out \
                --error=./logs/$experiment/%j.err \
                --wrap="source ~/.bashrc; activate bandit_pr; python src/baseline.py \
                    experiment=$experiment \
                    task=$task \
                    retriever=$retriever \
                    num_retrieve=$num_retrieve \
                    llm=$llm"
        done
    done
done
