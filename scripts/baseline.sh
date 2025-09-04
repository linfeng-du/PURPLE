#!/bin/bash

ARGS=$(getopt \
    --options "" \
    --long time:,llms:,tasks:,retrievers:,num_retrieve: \
    --name "$0" \
    -- "$@"
)
eval set -- "$ARGS"

while true; do
    case "$1" in
        --time) time="$2"; shift 2 ;;
        --llms) llms="$2"; shift 2 ;;
        --tasks) tasks="$2"; shift 2 ;;
        --retrievers) retrievers="$2"; shift 2 ;;
        --num_retrieve) num_retrieve="$2"; shift 2 ;;
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
            exp_name="$llm/$retriever-$num_retrieve/$task"
            mkdir -p ./logs/$exp_name
            sbatch \
                --job-name=$exp_name \
                --time=$time \
                --gpus-per-node=1 \
                --mem=64G \
                --output=./logs/$exp_name/%j.out \
                --error=./logs/$exp_name/%j.err \
                --wrap="source ~/.bashrc; activate bandit_pr; python src/baseline.py \
                    llm=$llm \
                    exp_name=$exp_name \
                    task=$task \
                    retriever=$retriever \
                    num_retrieve=$num_retrieve"
        done
    done
done
