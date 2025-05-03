#!/bin/bash

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            task=$2
            shift 2
            ;;
        --llm)
            llm=$2
            shift 2
            ;;
        --num_retrieve)
            num_retrieve=$2
            shift 2
            ;;
        *)
            echo "Invalid argument: $1"
            exit 1
            ;;
    esac
done

python src/train.py \
    experiment=$llm/bandit_pr-$num_retrieve/$task \
    task=$task \
    llm=$llm \
    num_retrieve=$num_retrieve
