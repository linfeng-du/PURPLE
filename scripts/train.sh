#!/bin/bash

ARGS=$(getopt \
    --options "" \
    --long time:,from_pretrained,llms:,tasks:,retrievers:,num_retrieve:,fuse_modes:,num_layers_list:,rewards: \
    --name "$0" \
    -- "$@"
)
eval set -- "$ARGS"

while true; do
    case "$1" in
        --time) time="$2"; shift 2 ;;
        --from_pretrained) from_pretrained=true; shift 1 ;;
        --llms) llms="$2"; shift 2 ;;
        --tasks) tasks="$2"; shift 2 ;;
        --retrievers) retrievers="$2"; shift 2 ;;
        --num_retrieve) num_retrieve="$2"; shift 2 ;;
        --fuse_modes) fuse_modes="$2"; shift 2 ;;
        --num_layers_list) num_layers_list="$2"; shift 2 ;;
        --rewards) rewards="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

: "${time:=24:00:00}"
: "${from_pretrained:=false}"
: "${llms:=phi-4-mini-instruct,llama-3-8b-instruct}"
: "${tasks:=LaMP-1,LaMP-2,LaMP-3,LaMP-4,LaMP-5,LaMP-7,LongLaMP-2,LongLaMP-3,LongLaMP-4}"
: "${retrievers:=bm25}"
: "${num_retrieve:=5}"
: "${fuse_modes:=cross_attn}"
: "${num_layers_list:=12}"
: "${rewards:=logp}"

IFS=',' read -ra llms <<< "$llms"
IFS=',' read -ra tasks <<< "$tasks"
IFS=',' read -ra retrievers <<< "$retrievers"
IFS=',' read -ra fuse_modes <<< "$fuse_modes"
IFS=',' read -ra num_layers_list <<< "$num_layers_list"
IFS=',' read -ra rewards <<< "$rewards"

for llm in ${llms[@]}; do
    for retriever in ${retrievers[@]}; do
        for fuse_mode in ${fuse_modes[@]}; do
            for num_layers in ${num_layers_list[@]}; do
                for reward in ${rewards[@]}; do
                    for task in ${tasks[@]}; do
                        exp_name="$llm/bandit_pr-$num_retrieve/$retriever/$fuse_mode-$num_layers-$reward/$task"
                        mkdir -p "./logs/$exp_name"
                        sbatch \
                            --job-name="$exp_name" \
                            --time="$time" \
                            --gres=gpu:h100:1 \
                            --mem=64G \
                            --output="./logs/$exp_name/%j.out" \
                            --error="./logs/$exp_name/%j.err" \
                            --wrap="source ~/.bashrc; activate bandit_pr; python src/train.py \
                                exp_name=\"$exp_name\" \
                                llm=\"$llm\" \
                                task=\"$task\" \
                                retriever=\"$retriever\" \
                                from_pretrained=\"$from_pretrained\" \
                                num_retrieve=\"$num_retrieve\" \
                                score_model.fuse_mode=\"$fuse_mode\" \
                                score_model.num_layers=\"$num_layers\" \
                                reinforce.reward=\"$reward\""
                    done
                done
            done
        done
    done
done
