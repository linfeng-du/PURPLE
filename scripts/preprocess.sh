#!/bin/bash

ARGS=$(
    getopt \
        --options "" \
        --long time:,tasks:,retrievers:,num_candidates: \
        --name "$0" \
        -- "$@"
)
eval set -- "$ARGS"

while true; do
    case "$1" in
        --time) time="$2"; shift 2 ;;
        --tasks) tasks="$2"; shift 2 ;;
        --retrievers) retrievers="$2"; shift 2 ;;
        --num_candidates) num_candidates="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

: "${time:=6:00:00}"
: "${tasks:=LaMP-1,LaMP-2,LaMP-3,LaMP-4,LaMP-5,LaMP-7,LongLaMP-2,LongLaMP-3,LongLaMP-4}"
: "${retrievers:=contriever,bm25}"
: "${num_candidates:=20}"

IFS=',' read -ra tasks <<< "$tasks"
IFS=',' read -ra retrievers <<< "$retrievers"


for retriever in ${retrievers[@]}; do
    for task in ${tasks[@]}; do
        exp_name="preprocess/$retriever-$num_candidates/$task"
        mkdir -p "./logs/$exp_name"
        sbatch \
            --job-name="$exp_name" \
            --time="$time" \
            --gres=gpu:h100:1 \
            --mem=128G \
            --output="./logs/$exp_name/%j.out" \
            --error="./logs/$exp_name/%j.err" \
            --wrap="$(
                echo -n "source ~/.bashrc; "
                echo -n "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
                echo -n "activate bandit_ramp; "
                echo -n "python src/process.py preprocess "
                echo -n "--task=\"$task\" "
                echo -n "--retriever=\"$retriever\" "
                echo -n "--num_candidates=\"$num_candidates\""
            )"
    done
done
