#!/bin/bash

ARGS=$(getopt --options "" --long time:,model:,port: --name "$0" -- "$@")
eval set -- "$ARGS"

while true; do
    case "$1" in
        --time) time="$2"; shift 2 ;;
        --model) model="$2"; shift 2 ;;
        --port) port="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

sbatch \
    --job-name="$model" \
    --time="$time" \
    --gpus-per-node=4 \
    --output="./logs/serve_llm/$model/%j.out" \
    --error="./logs/serve_llm/$model/%j.err" \
    --wrap="$(
        echo -n "source ~/.bashrc; "
        echo -n "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False; "
        echo -n "activate vllm; "
        echo -n "python -m vllm.entrypoints.openai.api_server "
        echo -n "--model \"$model\" "
        echo -n "--tensor_parallel_size 4 "
        echo -n "--host 0.0.0.0 "
        echo -n "--port \"$port\""
    )"
