#!/bin/bash

ARGS=$(getopt --options "" --long api,llms:,tasks:,num_retrieve: --name "$0" -- "$@")
eval set -- "$ARGS"

api=0

while true; do
    case "$1" in
        --api) api=1; shift ;;
        --llms) llms="$2"; shift 2 ;;
        --tasks) tasks="$2"; shift 2 ;;
        --num_retrieve) num_retrieve="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

IFS=',' read -ra llms <<< "$llms"
IFS=',' read -ra tasks <<< "$tasks"

for llm in ${llms[@]}; do
    for task in ${tasks[@]}; do
        experiment="$llm/bandit_pr-$num_retrieve/$task"
        sbatch \
            --job-name=$experiment \
            --time=48:0:0 \
            --gres=gpu:a100:1 \
            --mem=64G \
            --output=./logs/$experiment/%j.out \
            --error=./logs/$experiment/%j.err \
            --wrap="source ~/.bashrc; activate bandit_pr; python src/train.py \
                experiment=$experiment \
                task=$task \
                num_retrieve=$num_retrieve \
                llm=$llm"
    done
done
