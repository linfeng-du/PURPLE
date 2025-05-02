#!/bin/bash
#SBATCH --time=3:0:0
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=64G
#SBATCH --output=./logs/slurm/baseline/output_%j.txt
#SBATCH --error=./logs/slurm/baseline/error_%j.txt

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

source /home/linfeng/.bashrc
source /home/linfeng/projects/def-cpsmcgil/linfeng/environment/bandit_pr/bin/activate
python src/baseline.py \
    experiment=$llm/$retriever-$num_retrieve/$task \
    task=$task \
    llm=$llm \
    retriever=$retriever \
    num_retrieve=$num_retrieve
