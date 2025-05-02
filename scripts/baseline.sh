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
        --retriever)
            retriever=$2
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

python src/baseline.py \
    experiment=$llm/$retriever-$num_retrieve/$task \
    task=$task \
    llm=$llm \
    retriever=$retriever \
    num_retrieve=$num_retrieve
