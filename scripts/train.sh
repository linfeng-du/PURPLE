#!/bin/bash

ARGS=$(
    getopt \
        --options "" \
        --long $(
            echo -n "time:,tasks:,from_pretrained,"
            echo -n "llms:,retrievers:,num_candidates:,num_rerank:,"
            echo -n "fuse_modes:,num_layers_list:"
        ) \
        --name "$0" \
        -- "$@"
)
eval set -- "$ARGS"

while true; do
    case "$1" in
        --time) time="$2"; shift 2 ;;
        --tasks) tasks="$2"; shift 2 ;;
        --from_pretrained) from_pretrained=true; shift 1 ;;
        --llms) llms="$2"; shift 2 ;;
        --retrievers) retrievers="$2"; shift 2 ;;
        --num_candidates) num_candidates="$2"; shift 2 ;;
        --num_rerank) num_rerank="$2"; shift 2 ;;
        --fuse_modes) fuse_modes="$2"; shift 2 ;;
        --num_layers_list) num_layers_list="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

: "${time:=24:00:00}"
: "${tasks:=LaMP-1,LaMP-2,LaMP-3,LaMP-4,LaMP-5,LaMP-7,LongLaMP-2,LongLaMP-3,LongLaMP-4}"
: "${from_pretrained:=false}"
: "${llms:=phi-4-mini-instruct,llama-3-8b-instruct}"
: "${retrievers:=contriever}"
: "${num_candidates:=20}"
: "${num_rerank:=5}"
: "${fuse_modes:=cross_attn}"
: "${num_layers_list:=12}"

IFS=',' read -ra tasks <<< "$tasks"
IFS=',' read -ra llms <<< "$llms"
IFS=',' read -ra retrievers <<< "$retrievers"
IFS=',' read -ra fuse_modes <<< "$fuse_modes"
IFS=',' read -ra num_layers_list <<< "$num_layers_list"

for llm in ${llms[@]}; do
    for retriever in ${retrievers[@]}; do
        for fuse_mode in ${fuse_modes[@]}; do
            for num_layers in ${num_layers_list[@]}; do
                for task in ${tasks[@]}; do
                    exp_name=$(
                        echo -n "$llm/$retriever-$num_candidates/"
                        echo -n "bandit_ramp-$num_rerank/$fuse_mode-$num_layers/$task"
                    )
                    mkdir -p "./logs/$exp_name"
                    sbatch \
                        --job-name="$exp_name" \
                        --time="$time" \
                        --gres=gpu:h100:1 \
                        --mem=64G \
                        --output="./logs/$exp_name/%j.out" \
                        --error="./logs/$exp_name/%j.err" \
                        --wrap="$(
                            echo -n "source ~/.bashrc; "
                            echo -n "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
                            echo -n "activate bandit_ramp; "
                            echo -n "python src/train.py "
                            echo -n "exp_name=\"$exp_name\" "
                            echo -n "task=\"$task\" "
                            echo -n "from_pretrained=\"$from_pretrained\" "
                            echo -n "llm=\"$llm\" "
                            echo -n "retriever=\"$retriever\" "
                            echo -n "num_candidates=\"$num_candidates\" "
                            echo -n "num_rerank=\"$num_rerank\" "
                            echo -n "score_model.fuse_mode=\"$fuse_mode\" "
                            echo -n "score_model.num_layers=\"$num_layers\""
                        )"
                done
            done
        done
    done
done
