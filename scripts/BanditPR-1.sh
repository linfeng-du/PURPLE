#!/bin/bash

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)
            tasks=$2
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

IFS=',' read -r -a task_array <<< "$tasks"

for task in "${task_array[@]}"; do
    python src/train.py \
        experiment=BanditPR-1/$task \
        seed=42 \
        task=$task \
        n_retrieve=1
done
