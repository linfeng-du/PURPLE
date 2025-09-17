#!/bin/bash

ARGS=$(getopt --options "" --long time:,llm:,port: --name "$0" -- "$@")
eval set -- "$ARGS"

while true; do
    case "$1" in
        --time) time="$2"; shift 2 ;;
        --llm) llm="$2"; shift 2 ;;
        --port) port="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

sbatch \
    --job-name="$llm" \
    --time="$time" \
    --gres=gpu:h100:4 \
    --mem=256G \
    --output="./logs/serve_llm/$llm/%j.out" \
    --error="./logs/serve_llm/$llm/%j.err" \
    --wrap="$(
        echo -n "source ~/.bashrc; "
        echo -n "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False; "
        echo -n "activate bandit_ramp; "
        echo -n "python -m vllm.entrypoints.openai.api_server "
        echo -n "--model \"$llm\" "
        echo -n "--tensor_parallel_size 4 "
        echo -n "--host 0.0.0.0 "
        echo -n "--port \"$port\""
    )"
