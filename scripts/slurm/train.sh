#!/bin/bash
#SBATCH --time=48:0:0
#SBATCH --gres=gpu:v100l:2
#SBATCH --mem=64G
#SBATCH --output=./logs/slurm/train/output_%j.txt
#SBATCH --error=./logs/slurm/train/error_%j.txt

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
