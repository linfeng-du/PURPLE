#!/bin/bash

ARGS=$(getopt \
    --options "" \
    --long time:,llms:,tasks:,num_retrieve:,fuse_modes:,num_layers_list:,rewards:,losses: \
    --name "$0" \
    -- "$@"
)
eval set -- "$ARGS"

while true; do
    case "$1" in
        --time) time="$2"; shift 2 ;;
        --llms) llms="$2"; shift 2 ;;
        --tasks) tasks="$2"; shift 2 ;;
        --num_retrieve) num_retrieve="$2"; shift 2 ;;
        --fuse_modes) fuse_modes="$2"; shift 2 ;;
        --num_layers_list) num_layers_list="$2"; shift 2 ;;
        --rewards) rewards="$2"; shift 2 ;;
        --losses) losses="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

IFS=',' read -ra llms <<< "$llms"
IFS=',' read -ra tasks <<< "$tasks"
IFS=',' read -ra fuse_modes <<< "$fuse_modes"
IFS=',' read -ra num_layers_list <<< "$num_layers_list"
IFS=',' read -ra rewards <<< "$rewards"
IFS=',' read -ra losses <<< "$losses"

for llm in ${llms[@]}; do
    for task in ${tasks[@]}; do
        for fuse_mode in ${fuse_modes[@]}; do
            for num_layers in ${num_layers_list[@]}; do
                for reward in ${rewards[@]}; do
                    for loss in ${losses[@]}; do
                        exp_name="$llm/bandit_pr-$num_retrieve/$fuse_mode-$num_layers-$reward-$loss/$task"
                        mkdir -p ./logs/$exp_name
                        sbatch \
                            --job-name=$exp_name \
                            --time=$time \
                            --gpus-per-node=1 \
                            --output=./logs/$exp_name/%j.out \
                            --error=./logs/$exp_name/%j.err \
                            --wrap="source ~/.bashrc; activate bandit_pr; python src/train.py \
                                exp_name=$exp_name \
                                llm=$llm \
                                task=$task \
                                num_retrieve=$num_retrieve \
                                score_model.fuse_mode=$fuse_mode \
                                score_model.num_layers=$num_layers \
                                reinforce.reward=$reward \
                                reinforce.loss=$loss"
                    done
                done
            done
        done
    done
done
