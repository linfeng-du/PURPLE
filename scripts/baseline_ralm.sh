#!/bin/bash

ARGS=$(getopt \
    --options "" \
    --long time:,llms:,tasks:,ralms:,num_retrieve: \
    --name "$0" \
    -- "$@"
)
eval set -- "$ARGS"

while true; do
    case "$1" in
        --time) time="$2"; shift 2 ;;
        --llms) llms="$2"; shift 2 ;;
        --tasks) tasks="$2"; shift 2 ;;
        --ralms) ralm="$2"; shift 2 ;;
        --num_retrieve) num_retrieve="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

: "${time:=24:00:00}"
: "${llms:=phi-4-mini-instruct,llama-3-8b-instruct}"
: "${tasks:=LaMP-1,LaMP-2,LaMP-3,LaMP-4,LaMP-5,LaMP-7,LongLaMP-2,LongLaMP-3,LongLaMP-4}"
: "${ralms:=replug}"
: "${num_retrieve:=5}"

IFS=',' read -ra llms <<< "$llms"
IFS=',' read -ra tasks <<< "$tasks"

for llm in ${llms[@]}; do
    for task in ${tasks[@]}; do
        for ralm in ${ralms[@]}; do
            exp_name="$llm/$ralm-$num_retrieve/$task"
            mkdir -p "./logs/$exp_name"
            sbatch \
                --job-name="$exp_name" \
                --time="$time" \
                --gres=gpu:h100:1 \
                --mem=64G \
                --output="./logs/$exp_name/%j.out" \
                --error="./logs/$exp_name/%j.err" \
                --wrap="source ~/.bashrc; activate bandit_pr; python src/baseline_ralm.py \
                    \"$ralm\" \
                    --llm=\"$llm\" \
                    --task=\"$task\" \
                    --num_retrieve=\"$num_retrieve\""
        done
    done
done
